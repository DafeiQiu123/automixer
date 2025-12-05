import os
import sys
import math
import time
import json
import argparse
from typing import List, Tuple, Optional

import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 保证可以从项目根目录导入模块（例如 default_parameter_output.py）
_CUR_DIR = os.path.dirname(__file__)
_ROOT_DIR = os.path.abspath(os.path.join(_CUR_DIR, os.pardir))
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

from models.mert_encoder import MERTEncoder
from models.transformer_mixer import MixerTransformer
from default_parameter_output import extract_all_parameters, parse_song_info


def _list_audio_files(input_dir: str,
                      exts: Tuple[str, ...] = (".wav",)) -> List[str]:
    files = []
    for name in sorted(os.listdir(input_dir)):
        if name.lower().endswith(exts):
            files.append(os.path.abspath(os.path.join(input_dir, name)))
    return files


def build_pairs_from_dir(input_dir: str) -> List[Tuple[str, str]]:
    """
    从目录生成顺序相邻的 (A, B) 对。
    仅基于文件名排序。
    """
    files = _list_audio_files(input_dir)
    pairs: List[Tuple[str, str]] = []
    for i in range(len(files) - 1):
        pairs.append((files[i], files[i + 1]))
    return pairs


def build_adjacent_pairs_from_files(files: List[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for i in range(len(files) - 1):
        pairs.append((files[i], files[i + 1]))
    return pairs


def build_random_pairs_from_files(files: List[str],
                                  num_pairs: int,
                                  seed: int = 42) -> List[Tuple[str, str]]:
    """
    随机生成 (A, B) 对，允许重复抽样，但 A != B。
    """
    rng = random.Random(seed)
    files = list(files)
    if len(files) < 2:
        return []
    pairs: List[Tuple[str, str]] = []
    for _ in range(max(0, num_pairs)):
        a = rng.choice(files)
        b = rng.choice(files)
        # 确保 A != B
        tries = 0
        while b == a and tries < 10:
            b = rng.choice(files)
            tries += 1
        if a != b:
            pairs.append((a, b))
    return pairs


def list_all_unique_pairs(files: List[str]) -> List[Tuple[str, str]]:
    """
    生成所有不同的有序对 (A, B), A != B。上限 = N*(N-1)
    """
    files = list(files)
    out: List[Tuple[str, str]] = []
    for i, a in enumerate(files):
        for j, b in enumerate(files):
            if i == j:
                continue
            out.append((a, b))
    return out


def _load_pair_state(state_path: str) -> dict:
    if not os.path.isfile(state_path):
        return {"train": [], "valid": []}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"train": [], "valid": []}


def _save_pair_state(state_path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _choose_unique_pairs(all_pairs: List[Tuple[str, str]],
                         desired_n: int,
                         used_pairs: List[List[str]],
                         rng: random.Random) -> Tuple[List[Tuple[str, str]], List[List[str]]]:
    """
    从 all_pairs 中优先选择未使用的对（无放回），如不足再从已使用的中补齐（同样无重复）。
    used_pairs: JSON 里的列表形式 [[A,B], ...]
    返回: (chosen_pairs, newly_used_pairs_to_append)
    """
    used_set = set((a, b) for a, b in used_pairs)
    # 未使用与已使用池
    not_used = [p for p in all_pairs if p not in used_set]
    already_used = [p for p in all_pairs if p in used_set]

    # 去重安全
    desired_n = max(0, min(desired_n, len(all_pairs)))
    chosen: List[Tuple[str, str]] = []

    # 先从未使用中随机抽取
    if not_used:
        k = min(desired_n - len(chosen), len(not_used))
        if k > 0:
            chosen.extend(rng.sample(not_used, k))

    # 不足则从已使用池中补足（避免重复）
    if len(chosen) < desired_n and already_used:
        remaining = desired_n - len(chosen)
        pool = [p for p in already_used if p not in chosen]
        if pool:
            k2 = min(remaining, len(pool))
            if k2 > 0:
                chosen.extend(rng.sample(pool, k2))

    # 需要追加到 used 的是这次新用到且之前未用过的
    newly_used = [[a, b] for (a, b) in chosen if (a, b) not in used_set]
    return chosen, newly_used


# -----------------------------
# 目标参数归一化（Min-Max 到 [0,1]）
# 依据用户提供的典型范围
# 顺序: [hpf1, hpf2, lpf1, lpf2, eq_low, eq_mid, eq_high, duration_ratio]
# -----------------------------
TARGET_RANGES: List[Tuple[float, float]] = [
    (60.0, 200.0),     # hpf1
    (200.0, 600.0),    # hpf2
    (12000.0, 16000.0),# lpf1
    (3000.0, 8000.0),  # lpf2
    (0.8, 1.3),        # eq_low
    (1.0, 1.4),        # eq_mid
    (0.5, 1.2),        # eq_high
    (0.0, 1.0),        # duration_ratio
]


def _minmax_norm_vec(y_vec: np.ndarray) -> np.ndarray:
    y_norm = []
    for v, (mn, mx) in zip(y_vec.tolist(), TARGET_RANGES):
        if mx <= mn:
            y_norm.append(0.0)
        else:
            val = (float(v) - mn) / (mx - mn)
            y_norm.append(float(np.clip(val, 0.0, 1.0)))
    return np.asarray(y_norm, dtype=np.float32)


def denormalize_params(y_norm: np.ndarray,
                       ranges: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
    """
    将归一化到 [0,1] 的参数反归一化回真实数值。
    支持形状 (8,), (T, 8), (B, T, 8)。
    """
    if ranges is None:
        ranges = TARGET_RANGES
    y_norm = np.asarray(y_norm, dtype=np.float32)
    if y_norm.shape[-1] != len(ranges):
        raise ValueError(f"Last dim must be {len(ranges)}, got {y_norm.shape}")
    r = np.asarray(ranges, dtype=np.float32)  # (8, 2)
    mins = r[:, 0]
    spans = r[:, 1] - r[:, 0]
    # 广播到 y_norm 的形状
    reshape_shape = (1,) * (y_norm.ndim - 1) + (len(ranges),)
    mins_b = mins.reshape(reshape_shape)
    spans_b = spans.reshape(reshape_shape)
    y = mins_b + np.clip(y_norm, 0.0, 1.0) * spans_b
    return y.astype(np.float32)


class ABMixerDataset(Dataset):
    """
    针对 (A, B) 歌曲对：
      - 使用 extract_all_parameters 计算 8 维标签：
        [hpf1, hpf2, lpf1, lpf2, eq_low, eq_mid, eq_high, duration_ratio]
      - 使用 MERTEncoder 分别得到 A 段和 B 段的帧级 embedding（不再拼接额外 PE）
      - 模型内部用 DualBPMPositionalEncoding 注入节拍位置信息（需要 bpm_a, bpm_b）
      - 作为输入：x1 = H_A, x2 = H_B，时间维为 T
      - 作为输出：将 8 维标签在时间维复制为 (T, 8)
    """
    def __init__(self,
                 pairs: List[Tuple[str, str]],
                 mert_encoder: MERTEncoder,
                 target_frames: int = 200,
                 max_transition_seconds: int = 20):
        self.pairs = pairs
        self.encoder = mert_encoder
        self.target_frames = target_frames
        self.max_transition_seconds = max_transition_seconds
        self.sr_mert = self.encoder.target_sr

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_last_n_seconds(self, audio: torch.Tensor, seconds: float) -> torch.Tensor:
        if seconds is None or seconds <= 0:
            return audio
        num_samples = int(seconds * self.sr_mert)
        if num_samples >= audio.shape[0]:
            return audio
        return audio[-num_samples:]

    def _load_first_n_seconds(self, audio: torch.Tensor, seconds: float) -> torch.Tensor:
        if seconds is None or seconds <= 0:
            return audio
        num_samples = int(seconds * self.sr_mert)
        if num_samples >= audio.shape[0]:
            return audio
        return audio[:num_samples]

    def __getitem__(self, idx: int):
        path_A, path_B = self.pairs[idx]

        # 1) 先基于整首歌计算标签与过渡时长比例
        params = extract_all_parameters(
            path_A, path_B,
            sr=24000,
            max_transition_seconds=self.max_transition_seconds
        )
        # 目标向量（8 维）
        y_vec = np.array([
            params["hpf1"],
            params["hpf2"],
            params["lpf1"],
            params["lpf2"],
            params["eq_low"],
            params["eq_mid"],
            params["eq_high"],
            params["duration_ratio"],
        ], dtype=np.float32)

        # 2) 计算用于编码的片段时长（秒）
        duration_ratio = float(params["duration_ratio"])
        seg_seconds = max(0.1, duration_ratio * float(self.max_transition_seconds))

        # 3) 载入 16k 音频并截取 A 的尾段、B 的首段
        audio_A_full = self.encoder.load_audio(path_A, start_sec=None, end_sec=None)
        audio_B_full = self.encoder.load_audio(path_B, start_sec=None, end_sec=None)

        # audio_A_seg = self._load_last_n_seconds(audio_A_full, seg_seconds)
        # audio_B_seg = self._load_first_n_seconds(audio_B_full, seg_seconds)

        # 4) 提取 MERT 帧特征并重采样到 T 帧
        H_A = self.encoder.extract_embeddings(audio_A_full)  # (Ta, d)
        H_B = self.encoder.extract_embeddings(audio_B_full)  # (Tb, d)
        H_A = self.encoder.resample_frames(H_A, self.target_frames)  # (T, d)
        H_B = self.encoder.resample_frames(H_B, self.target_frames)  # (T, d)

        # 5) 组装输入 (T, d) - 仅 embedding
        X1 = H_A.astype(np.float32)
        X2 = H_B.astype(np.float32)

        # 6) 标签归一化到 [0,1]，并复制到每一帧 (T, 8)
        y_norm = _minmax_norm_vec(y_vec)  # (8,)
        Y = np.repeat(y_norm[None, :], self.target_frames, axis=0)  # (T, 8)

        X1_t = torch.from_numpy(X1)  # (T, d+2)
        X2_t = torch.from_numpy(X2)  # (T, d+2)
        Y_t = torch.from_numpy(Y)    # (T, 8)

        return X1_t, X2_t, Y_t, {
            "path_A": path_A,
            "path_B": path_B,
            "bpmA": float(params.get("bpmA", 120.0)),
            "bpmB": float(params.get("bpmB", 120.0)),
        }


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: str,
                    epoch_idx: Optional[int] = None,
                    total_epochs: Optional[int] = None,
                    iter_log: Optional[List[float]] = None) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    desc = "Train"
    if epoch_idx is not None and total_epochs is not None:
        desc = f"Train [{epoch_idx}/{total_epochs}]"

    for X1, X2, Y, bpmA, bpmB, _meta in tqdm(loader, desc=desc, dynamic_ncols=True, leave=False):
        X1 = X1.to(device)      # (B, T, d)
        X2 = X2.to(device)      # (B, T, d)
        Y = Y.to(device)        # (B, T, 8)
        bpmA = bpmA.to(device)  # (B,)
        bpmB = bpmB.to(device)  # (B,)

        optimizer.zero_grad(set_to_none=True)
        Y_pred, _ = model(X1, X2, bpmA, bpmB)     # (B, T, 8)
        loss = F.mse_loss(Y_pred, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = X1.size(0)
        loss_val = float(loss.item())
        total_loss += loss_val * bs
        total_count += bs
        if iter_log is not None:
            iter_log.append(loss_val)

    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: str,
             epoch_idx: Optional[int] = None,
             total_epochs: Optional[int] = None,
             iter_log: Optional[List[float]] = None) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    desc = "Valid"
    if epoch_idx is not None and total_epochs is not None:
        desc = f"Valid [{epoch_idx}/{total_epochs}]"

    for X1, X2, Y, bpmA, bpmB, _meta in tqdm(loader, desc=desc, dynamic_ncols=True, leave=False):
        X1 = X1.to(device)
        X2 = X2.to(device)
        Y = Y.to(device)
        bpmA = bpmA.to(device)
        bpmB = bpmB.to(device)
        Y_pred, _ = model(X1, X2, bpmA, bpmB)
        loss = F.mse_loss(Y_pred, Y)

        bs = X1.size(0)
        loss_val = float(loss.item())
        total_loss += loss_val * bs
        total_count += bs
        if iter_log is not None:
            iter_log.append(loss_val)

    return total_loss / max(1, total_count)


def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description="Train MixerTransformer from a directory of audio files.")
    parser.add_argument("--input_dir", type=str,
                        default=os.path.join(_ROOT_DIR, "data", "wav_dir"),
                        help="输入音频目录（包含 .wav）")
    parser.add_argument("--valid_ratio", type=float, default=0.1,
                        help="验证集占比 (0,1)")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="批大小")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--target_frames", type=int, default=200, help="每段帧数")
    parser.add_argument("--max_transition_seconds", type=int, default=20, help="用于参数分析的过渡最长秒数")
    parser.add_argument("--model_d_model", type=int, default=512, help="Transformer d_model")
    parser.add_argument("--model_nhead", type=int, default=8, help="多头注意力头数")
    parser.add_argument("--model_layers", type=int, default=4, help="Transformer 层数")
    parser.add_argument("--dsp_dim", type=int, default=8, help="输出 DSP 维度")
    parser.add_argument("--save_path", type=str,
                        default=os.path.join(_ROOT_DIR, "models", "mixer_checkpoint.pt"),
                        help="checkpoint 保存路径")
    parser.add_argument("--ckpt_dir", type=str,
                        default=os.path.join(_ROOT_DIR, "models", "checkpoints"),
                        help="保存所有 epoch 的 checkpoint 的目录")
    parser.add_argument("--plot_path", type=str,
                        default=os.path.join(_ROOT_DIR, "training_curve.png"),
                        help="训练曲线保存路径 (png)")
    parser.add_argument("--pairing_mode", type=str, default="random",
                        choices=["adjacent", "random"], help="构造 (A,B) 对的方式")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（用于 random pairing）")
    parser.add_argument("--num_pairs_train", type=int, default=None,
                        help="训练集随机对数（默认 len(train_files)-1）")
    parser.add_argument("--num_pairs_valid", type=int, default=None,
                        help="验证集随机对数（默认 len(valid_files)-1）")
    parser.add_argument("--use_pos_encoding", type=str, default="naive", choices=["naive", "bpm", "none"],
                        help="在模型中启用 BPM 位置编码，可选 naive, bpm, none")
    parser.add_argument("--pair_state_path", type=str, default=None,
                        help="跨次运行的 pair 状态文件（优先选择未使用对）。默认存到 ckpt_dir/pair_state.json")
    parser.add_argument("--same_song_ratio", type=float, default=0.15,
                        help="随机配对时，同一首歌片段对所占比例（0~1）")
    args = parser.parse_args()

    # 配置检查
    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    valid_ratio = float(args.valid_ratio)
    if not (0.0 < valid_ratio < 1.0):
        print(f"[WARN] invalid valid_ratio={valid_ratio}, fallback to 0.1")
        valid_ratio = 0.1

    target_frames = int(args.target_frames)
    max_transition_seconds = int(args.max_transition_seconds)
    batch_size = int(args.batch_size)
    num_workers = int(args.num_workers)
    epochs = int(args.epochs)
    lr = float(args.lr)
    d_model = int(args.model_d_model)
    nhead = int(args.model_nhead)
    num_layers = int(args.model_layers)
    dsp_dim = int(args.dsp_dim)
    ckpt_path = args.save_path
    ckpt_dir = args.ckpt_dir
    plot_path = args.plot_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 基于文件列表划分训练/验证
    all_files = _list_audio_files(input_dir)
    if len(all_files) < 2:
        raise RuntimeError(f"Not enough files in {input_dir}. Need at least 2 wavs.")
    split_idx = int(round((1.0 - valid_ratio) * len(all_files)))
    split_idx = min(max(split_idx, 1), len(all_files) - 1)
    train_files = all_files[:split_idx]
    valid_files = all_files[split_idx:]

    if args.pairing_mode == "adjacent":
        train_pairs = build_adjacent_pairs_from_files(train_files)
        valid_pairs = build_adjacent_pairs_from_files(valid_files)
        available_train = len(train_pairs)
        available_valid = len(valid_pairs)
        newly_used_train = []
        newly_used_valid = []
    else:
        # 随机配对：控制同曲对比例 same_song_ratio，且优先选未使用的对
        def _safe_parse(path: str) -> Tuple[str, int]:
            try:
                sid, part = parse_song_info(path)
                return str(sid), int(part)
            except Exception:
                base = os.path.splitext(os.path.basename(path))[0]
                if "_" in base:
                    head, tail = base.rsplit("_", 1)
                    try:
                        return head, int(tail)
                    except Exception:
                        pass
                return base, -1

        def _split_pools(files: List[str]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
            """
            将所有有序对按是否来自同一首歌进行划分：
              - same_pool: id 相同且 part 不同（如 *_1.wav → *_2.wav 或反向）
              - diff_pool: 其余对（不同歌曲）
            """
            same_pool: List[Tuple[str, str]] = []
            diff_pool: List[Tuple[str, str]] = []
            parsed = [(_safe_parse(p), p) for p in files]  # [((id, part), path)]
            for i, ((idA, partA), a) in enumerate(parsed):
                for j, ((idB, partB), b) in enumerate(parsed):
                    if i == j:
                        continue
                    if idA == idB and partA != partB:
                        same_pool.append((a, b))
                    else:
                        diff_pool.append((a, b))
            return same_pool, diff_pool

        same_train, diff_train = _split_pools(train_files)
        same_valid, diff_valid = _split_pools(valid_files)

        max_train = len(same_train) + len(diff_train)
        max_valid = len(same_valid) + len(diff_valid)

        n_train_req = args.num_pairs_train if args.num_pairs_train is not None else max(1, len(train_files) - 1)
        n_valid_req = args.num_pairs_valid if args.num_pairs_valid is not None else max(1, len(valid_files) - 1)
        n_train = min(max_train, max(0, n_train_req))
        n_valid = min(max_valid, max(0, n_valid_req))

        state_path = args.pair_state_path or os.path.join(os.path.abspath(args.ckpt_dir), "pair_state.json")
        state = _load_pair_state(state_path)
        used_train_pairs: List[List[str]] = state.get("train", [])
        used_valid_pairs: List[List[str]] = state.get("valid", [])

        rng_train = random.Random(args.seed)
        rng_valid = random.Random(args.seed + 1)

        # 目标同曲比例
        same_ratio = float(np.clip(args.same_song_ratio, 0.0, 1.0))
        n_same_train = min(len(same_train), int(round(same_ratio * n_train)))
        n_same_valid = min(len(same_valid), int(round(same_ratio * n_valid)))

        # 先选同曲（未用过优先），再用不同曲补齐
        train_same_sel, train_same_new = _choose_unique_pairs(same_train, n_same_train, used_train_pairs, rng_train)
        used_train_after_same = used_train_pairs + train_same_new
        train_diff_needed = n_train - len(train_same_sel)
        train_diff_sel, train_diff_new = _choose_unique_pairs(diff_train, train_diff_needed, used_train_after_same, rng_train)
        train_pairs = train_same_sel + train_diff_sel
        newly_used_train = train_same_new + train_diff_new

        valid_same_sel, valid_same_new = _choose_unique_pairs(same_valid, n_same_valid, used_valid_pairs, rng_valid)
        used_valid_after_same = used_valid_pairs + valid_same_new
        valid_diff_needed = n_valid - len(valid_same_sel)
        valid_diff_sel, valid_diff_new = _choose_unique_pairs(diff_valid, valid_diff_needed, used_valid_after_same, rng_valid)
        valid_pairs = valid_same_sel + valid_diff_sel
        newly_used_valid = valid_same_new + valid_diff_new

        # 更新状态
        state["train"] = used_train_pairs + newly_used_train
        state["valid"] = used_valid_pairs + newly_used_valid
        _save_pair_state(state_path, state)
        available_train = max_train
        available_valid = max_valid

    print(f"[Data] input_dir={input_dir}")
    print(f"[Data] pairing_mode={args.pairing_mode}, seed={args.seed}")
    if args.pairing_mode == "random":
        print(f"[Data] same_song_ratio={float(np.clip(args.same_song_ratio, 0.0, 1.0)):.3f}")
    print(f"[Data] #train files={len(train_files)}, #valid files={len(valid_files)}")
    print(f"[Data] #train pairs={len(train_pairs)} (available unique={available_train}), "
          f"#valid pairs={len(valid_pairs)} (available unique={available_valid}) "
          f"(valid_ratio={valid_ratio:.3f})")
    print(f"[Config] target_frames={target_frames}, max_transition_seconds={max_transition_seconds}")
    print(f"[Config] epochs={epochs}, batch_size={batch_size}, num_workers={num_workers}, lr={lr}")
    print(f"[Model] d_model={d_model}, nhead={nhead}, layers={num_layers}, dsp_dim={dsp_dim}, use_pos_encoding={args.use_pos_encoding}")
    print(f"[Save] ckpt_path(best)={ckpt_path}")
    print(f"[Save] ckpt_dir(all epochs)={ckpt_dir}")

    # 编码器与模型
    mert = MERTEncoder(model_name="m-a-p/MERT-v1-95M")
    d_mert = int(mert.embedding_dim)
    in_dim_total = 2 * d_mert        # [H_A] + [H_B]

    model = MixerTransformer(
        in_dim=in_dim_total,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dsp_dim=dsp_dim,
        positional_encoding=args.use_pos_encoding
    ).to(device)

    # 数据集
    train_ds = ABMixerDataset(
        pairs=train_pairs,
        mert_encoder=mert,
        target_frames=target_frames,
        max_transition_seconds=max_transition_seconds
    )
    valid_ds = ABMixerDataset(
        pairs=valid_pairs,
        mert_encoder=mert,
        target_frames=target_frames,
        max_transition_seconds=max_transition_seconds
    )

    # DataLoader
    def _collate(batch):
        X1_list, X2_list, Y_list, bpmA_list, bpmB_list, M_list = [], [], [], [], [], []
        for X1, X2, Y, M in batch:
            X1_list.append(X1)
            X2_list.append(X2)
            Y_list.append(Y)
            bpmA_list.append(M["bpmA"])
            bpmB_list.append(M["bpmB"])
            M_list.append(M)
        X1_t = torch.stack(X1_list, dim=0)
        X2_t = torch.stack(X2_list, dim=0)
        Y_t = torch.stack(Y_list, dim=0)
        bpmA_t = torch.tensor(bpmA_list, dtype=torch.float32)
        bpmB_t = torch.tensor(bpmB_list, dtype=torch.float32)
        return X1_t, X2_t, Y_t, bpmA_t, bpmB_t, M_list

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=_collate
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=_collate
    )

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_valid = float("inf")
    train_losses: List[float] = []
    valid_losses: List[float] = []
    # 逐迭代曲线（跨 epoch 累积）
    train_iter_losses: List[float] = []
    valid_iter_losses: List[float] = []
    # 确保 checkpoint 目录存在
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in tqdm(range(1, epochs + 1), total=epochs, desc="Epochs", dynamic_ncols=True):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     epoch_idx=epoch, total_epochs=epochs,
                                     iter_log=train_iter_losses)
        valid_loss = evaluate(model, valid_loader, device,
                              epoch_idx=epoch, total_epochs=epochs,
                              iter_log=valid_iter_losses)
        t1 = time.time()
        tqdm.write(f"[Epoch {epoch:02d}] train_loss={train_loss:.6f} "
                   f"valid_loss={valid_loss:.6f} time={t1 - t0:.1f}s")

        train_losses.append(float(train_loss))
        valid_losses.append(float(valid_loss))

        # 准备 checkpoint payload
        payload = {
            "model_state": model.state_dict(),
            "config": {
                "in_dim": in_dim_total,
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": num_layers,
                "dsp_dim": dsp_dim,
                "positional_encoding": bool(args.use_pos_encoding),
                "target_frames": target_frames,
                "target_norm": {
                    "type": "minmax",
                    "ranges": TARGET_RANGES,
                },
            }
        }
        # 保存本 epoch 的 checkpoint（全部保留）
        epoch_ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch:03d}.pt")
        torch.save(payload, epoch_ckpt_path)
        tqdm.write(f"[Checkpoint] Saved epoch {epoch:03d} to: {epoch_ckpt_path}")

        if valid_loss < best_valid:
            best_valid = valid_loss
            # 另存一份 best 到指定路径和 ckpt_dir/best.pt
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(payload, ckpt_path)
            torch.save(payload, best_path)
            tqdm.write(f"[Checkpoint] Saved BEST to: {ckpt_path} and {best_path}")

    # 绘制训练曲线
    try:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        # 以迭代为单位的曲线
        iters_axis_train = list(range(1, len(train_iter_losses) + 1))
        iters_axis_valid = list(range(1, len(valid_iter_losses) + 1))
        plt.figure(figsize=(8, 5))
        plt.plot(iters_axis_train, train_iter_losses, label="train_iter_loss")
        plt.plot(iters_axis_valid, valid_iter_losses, label="valid_iter_loss")
        plt.xlabel("Iteration")
        plt.ylabel("MSE Loss")
        plt.title("Training Curve (per-iteration)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        tqdm.write(f"[Plot] Training curve saved to: {plot_path}")
    except Exception as e:
        tqdm.write(f"[Plot][WARN] Failed to save training curve: {e}")

if __name__ == "__main__":
    main()


