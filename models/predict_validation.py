import os
import sys
import json
import argparse
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 允许从项目根目录导入
_CUR_DIR = os.path.dirname(__file__)
_ROOT_DIR = os.path.abspath(os.path.join(_CUR_DIR, os.pardir))
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

from models.data_pipeline import (  # type: ignore
    ABMixerDataset,
    build_pairs_from_dir,
    denormalize_params,
    TARGET_RANGES,
)
from models.mert_encoder import MERTEncoder  # type: ignore
from models.transformer_mixer import MixerTransformer  # type: ignore


def load_checkpoint(ckpt_path: str, device: str) -> Tuple[torch.nn.Module, dict]:
    payload = torch.load(ckpt_path, map_location=device)
    cfg = payload["config"]
    model = MixerTransformer(
        in_dim=int(cfg["in_dim"]),
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_layers=int(cfg["num_layers"]),
        dsp_dim=int(cfg["dsp_dim"]),
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
    model, cfg = load_checkpoint(ckpt_file, device)

    # 构造验证对（与训练相同的切分方式）
    all_pairs = build_pairs_from_dir(input_dir)
    if len(all_pairs) < 2:
        raise RuntimeError(f"Not enough pairs found in {input_dir}. Need at least 2 files.")
    split = int(round((1.0 - valid_ratio) * len(all_pairs)))
    split = min(max(split, 1), len(all_pairs) - 1)
    valid_pairs = all_pairs[split:]
    print(f"[Data] #valid pairs={len(valid_pairs)}")

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


