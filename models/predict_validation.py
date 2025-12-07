import os
import sys
import json
import argparse
from typing import List, Tuple, Optional

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

# 允许从项目根目录导入
_CUR_DIR = os.path.dirname(__file__)
_ROOT_DIR = os.path.abspath(os.path.join(_CUR_DIR, os.pardir))
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

from models.data_pipeline import (  # type: ignore
    ABMixerDataset,
    _list_audio_files,
    build_adjacent_pairs_from_files,
    build_random_pairs_from_files,
    denormalize_params,
    TARGET_RANGES,
)
from models.mert_encoder import MERTEncoder  # type: ignore
from models.transformer_mixer import MixerTransformer  # type: ignore
from default_parameter_output import extract_all_parameters  # type: ignore


def _infer_pe_from_path(ckpt_path: str) -> Optional[str]:
    p = ckpt_path.lower()
    if "pe_bpm" in p:
        return "bpm"
    if "pe_naive" in p:
        return "naive"
    if "pe_none" in p or "no_pe" in p:
        return "none"
    return None


def _audio_duration_seconds(path: str) -> float:
    """
    使用 torchaudio.info 获取音频时长（秒）。
    兼容 wav/mp3 等常见格式。
    """
    try:
        info = torchaudio.info(path)
        return float(info.num_frames) / max(1, int(info.sample_rate))
    except Exception:
        # 回退：直接载入音频推断时长
        try:
            wav, sr = torchaudio.load(path)
            return float(wav.shape[-1]) / max(1, int(sr))
        except Exception:
            return 0.0


def _median_ref_seconds_from_trimmed_split() -> Optional[float]:
    """
    扫描 data/wav_dir_trimmed_split 下的 *.wav 文件，
    返回这些切片的“中位数时长（秒）”作为参考裁剪长度。
    若目录不存在或无有效文件则返回 None。
    """
    ref_dir = os.path.join(_ROOT_DIR, "data", "wav_dir_trimmed_split")
    if not os.path.isdir(ref_dir):
        return None
    try:
        seconds: List[float] = []
        for name in sorted(os.listdir(ref_dir)):
            if name.lower().endswith(".wav"):
                dur = _audio_duration_seconds(os.path.join(ref_dir, name))
                if dur > 0:
                    seconds.append(dur)
        if seconds:
            return float(np.median(seconds))
        return None
    except Exception:
        return None


def load_checkpoint(
    ckpt_path: str,
    device: str,
    positional_encoding_override: Optional[str] = None,
    d_model_override: Optional[int] = None,
    nhead_override: Optional[int] = None,
    num_layers_override: Optional[int] = None,
    dsp_dim_override: Optional[int] = None,
) -> Tuple[torch.nn.Module, dict]:
    payload = torch.load(ckpt_path, map_location=device)
    cfg = payload["config"]
    # 兼容历史 ckpt 中将 positional_encoding 误存为 bool 的情况
    if positional_encoding_override is not None:
        pe_type = str(positional_encoding_override)
    else:
        pe_cfg = cfg.get("positional_encoding", "naive")
        if isinstance(pe_cfg, bool):
            pe_from_name = _infer_pe_from_path(ckpt_path)
            if pe_from_name is not None:
                pe_type = pe_from_name
            else:
                pe_type = "naive" if pe_cfg else "none"
        elif isinstance(pe_cfg, str):
            pe_type = pe_cfg
        else:
            pe_type = "naive"

    model = MixerTransformer(
        in_dim=int(cfg["in_dim"]),
        d_model=int(d_model_override) if d_model_override is not None else int(cfg["d_model"]),
        nhead=int(nhead_override) if nhead_override is not None else int(cfg["nhead"]),
        num_layers=int(num_layers_override) if num_layers_override is not None else int(cfg["num_layers"]),
        dsp_dim=int(dsp_dim_override) if dsp_dim_override is not None else int(cfg["dsp_dim"]),
        positional_encoding=pe_type,
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, cfg


def find_ckpt(ckpt_path: Optional[str], ckpt_dir: Optional[str]) -> str:
    if ckpt_path and os.path.isfile(ckpt_path):
        return ckpt_path
    if ckpt_dir:
        best = os.path.join(ckpt_dir, "best.pt")
        if os.path.isfile(best):
            return best
        # fallback: pick latest checkpoint_epoch_*.pt
        if os.path.isdir(ckpt_dir):
            cands = sorted(
                [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)
                 if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
            )
            if cands:
                return cands[-1]
    # project default single-file best
    default_best = os.path.join(_ROOT_DIR, "models", "mixer_checkpoint.pt")
    if os.path.isfile(default_best):
        return default_best
    raise FileNotFoundError("No checkpoint found. Please provide --ckpt_path or --ckpt_dir.")


@torch.no_grad()
def predict_pair(
    path_A: str,
    path_B: str,
    ckpt: str,
    *,
    target_frames: Optional[int] = None,
    max_transition_seconds: int = 20,
    save_time_series: bool = False,
    output_json_path: Optional[str] = None,
    override_pe: Optional[str] = None,
    override_d_model: Optional[int] = None,
    override_nhead: Optional[int] = None,
    override_num_layers: Optional[int] = None,
    override_dsp_dim: Optional[int] = None,
) -> dict:
    """
    对单一 (A, B) 曲目对做推理。
    - 裁剪策略：第一首歌取后半段，第二首歌取前半段（各自一半时长）。
    - 参照 data_pipeline 的处理：MERT → 重采样到 T 帧 → 模型预测 → 反归一化。
    - 从 checkpoint 中恢复模型架构和 PE 类型（必要时从路径名推断）。
    返回包含 8 个参数和元数据的字典；可选保存到 JSON，并可保存逐帧序列 .npy。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_file = find_ckpt(ckpt, None)
    model, cfg = load_checkpoint(
        ckpt_file,
        device,
        positional_encoding_override=override_pe,
        d_model_override=override_d_model,
        nhead_override=override_nhead,
        num_layers_override=override_num_layers,
        dsp_dim_override=override_dsp_dim,
    )

    # 反归一化范围
    norm_cfg = cfg.get("target_norm", {"type": "minmax", "ranges": TARGET_RANGES})
    ranges = norm_cfg.get("ranges", TARGET_RANGES)

    # 编码器
    mert = MERTEncoder(model_name="m-a-p/MERT-v1-95M")
    T = int(target_frames) if target_frames is not None else int(cfg.get("target_frames", 200))

    # 先分析参数（得到 bpmA/bpmB 与 duration_ratio）
    params_ana = extract_all_parameters(
        path_A,
        path_B,
        sr=int(mert.target_sr),
        max_transition_seconds=int(max_transition_seconds),
    )
    bpmA = float(params_ana.get("bpmA", 120.0))
    bpmB = float(params_ana.get("bpmB", 120.0))
    duration_ratio = float(params_ana.get("duration_ratio", 0.5))

    # 裁剪边界（单位：秒）：A=后半段；B=前半段
    dur_A = _audio_duration_seconds(path_A)
    dur_B = _audio_duration_seconds(path_B)
    if dur_A > 0:
        start_A = max(0.0, 0.5 * dur_A)
        end_A = dur_A
    else:
        start_A, end_A = 0.0, None
    if dur_B > 0:
        start_B = 0.0
        end_B = max(0.1, 0.5 * dur_B)
    else:
        start_B, end_B = 0.0, None

    # 提取帧特征并重采样到 T
    H_A = mert.encode_segment(path_A, start_A, end_A, T)  # (T, d)
    H_B = mert.encode_segment(path_B, start_B, end_B, T)  # (T, d)

    # 转 Tensor，跑模型
    X1 = torch.from_numpy(H_A).unsqueeze(0).to(device)  # (1, T, d)
    X2 = torch.from_numpy(H_B).unsqueeze(0).to(device)  # (1, T, d)
    bpmA_t = torch.tensor([bpmA], dtype=torch.float32, device=device)
    bpmB_t = torch.tensor([bpmB], dtype=torch.float32, device=device)

    y_hat_norm, _ = model(X1, X2, bpmA_t, bpmB_t)  # (1, T, 8)
    y_hat_norm_np = y_hat_norm.detach().cpu().numpy()[0]  # (T, 8)

    # 聚合为单个 8 维向量，并反归一化
    y_mean = y_hat_norm_np.mean(axis=0)  # (8,)
    y_denorm = denormalize_params(y_mean, ranges=ranges)

    keys = ["hpf1", "hpf2", "lpf1", "lpf2", "eq_low", "eq_mid", "eq_high", "duration_ratio"]
    result = {k: float(v) for k, v in zip(keys, y_denorm.tolist())}
    result["path_A"] = path_A
    result["path_B"] = path_B
    result["bpmA"] = bpmA
    result["bpmB"] = bpmB
    if dur_A > 0 and dur_B > 0:
        result["segment_A_seconds"] = float(max(0.0, (end_A if end_A is not None else dur_A) - (start_A or 0.0)))
        result["segment_B_seconds"] = float(max(0.0, (end_B if end_B is not None else dur_B) - (start_B or 0.0)))

    # 可选：保存 JSON
    if output_json_path is not None:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # 可选：保存逐帧
    if save_time_series and output_json_path is not None:
        ts_path = os.path.splitext(output_json_path)[0] + "_series.npy"
        y_hat_ts = denormalize_params(y_hat_norm_np, ranges=ranges)  # (T, 8)
        np.save(ts_path, y_hat_ts)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run inference on validation split and save per-pair results.")
    parser.add_argument("--input_dir", type=str,
                        default=os.path.join(_ROOT_DIR, "data", "wav_dir"),
                        help="输入音频目录（包含 .wav）")
    parser.add_argument("--valid_ratio", type=float, default=0.1,
                        help="验证集占比 (0,1)")
    parser.add_argument("--target_frames", type=int, default=None,
                        help="覆盖 checkpoint 中保存的 target_frames（可选）")
    parser.add_argument("--max_transition_seconds", type=int, default=20,
                        help="与训练一致的过渡最长秒数")
    parser.add_argument("--ckpt_path", type=str, default=None, help="模型 checkpoint 路径（可选）")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="模型 checkpoint 目录（可选）")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(_ROOT_DIR, "models", "predictions"),
                        help="每对音乐的预测 JSON 输出目录")
    parser.add_argument("--batch_size", type=int, default=1, help="推理 batch size（建议 1）")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--save_time_series", action="store_true", help="是否保存逐帧预测到 .npy")
    parser.add_argument("--pairing_mode", type=str, default="random",
                        choices=["adjacent", "random"], help="构造 (A,B) 对的方式（验证集）")
    parser.add_argument("--num_pairs_valid", type=int, default=None,
                        help="验证集随机对数（默认 len(valid_files)-1）")
    parser.add_argument("--seed", type=int, default=123, help="随机种子（用于 random pairing）")
    # 结构覆盖选项（如需与 checkpoint 不一致或修复历史 ckpt 中 PE 配置）
    parser.add_argument("--pe_type", type=str, default=None, choices=["naive", "bpm", "none"],
                        help="覆盖使用的 positional encoding 类型（默认沿用 ckpt 配置，或从路径推断）")
    parser.add_argument("--d_model_override", type=int, default=None, help="覆盖 d_model")
    parser.add_argument("--nhead_override", type=int, default=None, help="覆盖 nhead")
    parser.add_argument("--num_layers_override", type=int, default=None, help="覆盖 num_layers")
    parser.add_argument("--dsp_dim_override", type=int, default=None, help="覆盖 dsp_dim")
    args = parser.parse_args()

    # 路径与设备
    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    valid_ratio = float(args.valid_ratio)
    if not (0.0 < valid_ratio < 1.0):
        valid_ratio = 0.1
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpoint
    ckpt_file = find_ckpt(args.ckpt_path, args.ckpt_dir)
    print(f"[Load] checkpoint: {ckpt_file}")
    model, cfg = load_checkpoint(
        ckpt_file,
        device,
        positional_encoding_override=args.pe_type,
        d_model_override=args.d_model_override,
        nhead_override=args.nhead_override,
        num_layers_override=args.num_layers_override,
        dsp_dim_override=args.dsp_dim_override,
    )

    # 构造验证对（与训练切分规则一致）
    all_files = _list_audio_files(input_dir)
    if len(all_files) < 2:
        raise RuntimeError(f"Not enough files in {input_dir}. Need at least 2 wavs.")
    split_idx = int(round((1.0 - valid_ratio) * len(all_files)))
    split_idx = min(max(split_idx, 1), len(all_files) - 1)
    valid_files = all_files[split_idx:]
    if args.pairing_mode == "adjacent":
        valid_pairs = build_adjacent_pairs_from_files(valid_files)
    else:
        n_valid = args.num_pairs_valid if args.num_pairs_valid is not None else max(1, len(valid_files) - 1)
        valid_pairs = build_random_pairs_from_files(valid_files, n_valid, seed=args.seed)
    print(f"[Data] pairing_mode={args.pairing_mode}, #valid files={len(valid_files)}, #valid pairs={len(valid_pairs)}")

    # 编码器与数据集
    mert = MERTEncoder(model_name="m-a-p/MERT-v1-95M")
    target_frames = int(args.target_frames) if args.target_frames is not None else int(cfg.get("target_frames", 200))
    max_transition_seconds = int(args.max_transition_seconds)
    ds = ABMixerDataset(
        pairs=valid_pairs,
        mert_encoder=mert,
        target_frames=target_frames,
        max_transition_seconds=max_transition_seconds
    )

    # collate：返回 X1, X2, bpmA, bpmB, meta（丢弃 Y）
    def _collate_eval(batch):
        X1_list, X2_list, bpmA_list, bpmB_list, M_list = [], [], [], [], []
        for X1, X2, _Y, M in batch:
            X1_list.append(X1)
            X2_list.append(X2)
            bpmA_list.append(M["bpmA"])
            bpmB_list.append(M["bpmB"])
            M_list.append(M)
        X1_t = torch.stack(X1_list, dim=0)
        X2_t = torch.stack(X2_list, dim=0)
        bpmA_t = torch.tensor(bpmA_list, dtype=torch.float32)
        bpmB_t = torch.tensor(bpmB_list, dtype=torch.float32)
        return X1_t, X2_t, bpmA_t, bpmB_t, M_list

    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device == "cuda"),
        collate_fn=_collate_eval
    )

    # 反归一化范围
    norm_cfg = cfg.get("target_norm", {"type": "minmax", "ranges": TARGET_RANGES})
    ranges = norm_cfg.get("ranges", TARGET_RANGES)

    model.eval()
    with torch.no_grad():
        for X1, X2, bpmA, bpmB, metas in tqdm(loader, desc="Infer", dynamic_ncols=True):
            X1 = X1.to(device)
            X2 = X2.to(device)
            bpmA = bpmA.to(device)
            bpmB = bpmB.to(device)
            y_hat_norm, _ = model(X1, X2, bpmA, bpmB)  # (B, T, 8)
            y_hat_norm_np = y_hat_norm.detach().cpu().numpy()  # (B, T, 8)

            # 对每个样本分别保存
            B = y_hat_norm_np.shape[0]
            for i in range(B):
                meta = metas[i]
                A_path = meta["path_A"]
                B_path = meta["path_B"]
                baseA = os.path.splitext(os.path.basename(A_path))[0]
                baseB = os.path.splitext(os.path.basename(B_path))[0]
                stem = f"{baseA}__{baseB}"

                # 平均得到单个 8 维向量（也可以用中位数）
                y_hat_mean = y_hat_norm_np[i].mean(axis=0)  # (8,)
                y_hat = denormalize_params(y_hat_mean, ranges=ranges)  # (8,)

                keys = ["hpf1", "hpf2", "lpf1", "lpf2", "eq_low", "eq_mid", "eq_high", "duration_ratio"]
                result = {k: float(v) for k, v in zip(keys, y_hat.tolist())}
                result["path_A"] = A_path
                result["path_B"] = B_path
                result["bpmA"] = float(meta["bpmA"])
                result["bpmB"] = float(meta["bpmB"])

                # 计算并写入 GT（未归一化）
                try:
                    gt_params = extract_all_parameters(
                        A_path,
                        B_path,
                        sr=int(mert.target_sr),
                        max_transition_seconds=max_transition_seconds,
                    )
                    gt = {
                        "hpf1": float(gt_params["hpf1"]),
                        "hpf2": float(gt_params["hpf2"]),
                        "lpf1": float(gt_params["lpf1"]),
                        "lpf2": float(gt_params["lpf2"]),
                        "eq_low": float(gt_params["eq_low"]),
                        "eq_mid": float(gt_params["eq_mid"]),
                        "eq_high": float(gt_params["eq_high"]),
                        "duration_ratio": float(gt_params["duration_ratio"]),
                    }
                    result["gt"] = gt
                except Exception as e:
                    # 失败时仅记录错误，不阻断其他对的保存
                    result["gt_error"] = str(e)

                json_path = os.path.join(output_dir, f"{stem}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                if args.save_time_series:
                    ts_path = os.path.join(output_dir, f"{stem}_series.npy")
                    # 也保存反归一化逐帧预测
                    y_hat_ts = denormalize_params(y_hat_norm_np[i], ranges=ranges)  # (T, 8)
                    np.save(ts_path, y_hat_ts)

                tqdm.write(f"[Saved] {json_path}")


if __name__ == "__main__":
    main()


