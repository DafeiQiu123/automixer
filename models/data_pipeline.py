import os
import sys
import math
import time
import json
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 保证可以从项目根目录导入模块（例如 default_parameter_output.py）
_CUR_DIR = os.path.dirname(__file__)
_ROOT_DIR = os.path.abspath(os.path.join(_CUR_DIR, os.pardir))
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

from models.mert_encoder import MERTEncoder
from models.bpm_encoding import bpm_position_encoding
from models.transformer_mixer import MixerTransformer
from default_parameter_output import extract_all_parameters


def _is_probably_root_line(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    # 认为不含制表符并且看起来像路径的第一行是根目录
    return ("\t" not in line) and ("/" in line or "\\" in line)


def _parse_manifest(tsv_path: str,
                    override_root: Optional[str] = None) -> List[str]:
    """
    支持两种 manifest 结构：
    1) 第一行是根目录，后续行: "filename<TAB>length"
    2) 每一行是完整路径或 "filename<TAB>length"
    返回：音频文件的绝对路径列表（按顺序）
    """
    with open(tsv_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if not lines:
        return []

    root_from_file = None
    start_idx = 0
    if _is_probably_root_line(lines[0]):
        root_from_file = lines[0]
        start_idx = 1

    audio_paths: List[str] = []
    for ln in lines[start_idx:]:
        parts = ln.split("\t")
        # 处理 "path\tlength" 或 "filename\tlength" 或 单列完整路径
        if len(parts) == 1:
            item = parts[0]
            if os.path.isabs(item):
                path = item
            else:
                base_root = override_root or root_from_file or os.path.dirname(tsv_path)
                path = os.path.join(base_root, item)
        else:
            item = parts[0]
            # 如果 item 看起来是绝对路径，直接用；否则拼接根目录
            if os.path.isabs(item):
                path = item
            else:
                base_root = override_root or root_from_file or os.path.dirname(tsv_path)
                path = os.path.join(base_root, item)
        audio_paths.append(os.path.abspath(path))

    return audio_paths


def build_pairs_from_manifest(tsv_path: str,
                              override_root: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    从 manifest 生成顺序相邻的 (A, B) 对。
    """
    files = _parse_manifest(tsv_path, override_root)
    pairs: List[Tuple[str, str]] = []
    for i in range(len(files) - 1):
        pairs.append((files[i], files[i + 1]))
    return pairs


class ABMixerDataset(Dataset):
    """
    针对 (A, B) 歌曲对：
      - 使用 extract_all_parameters 计算 8 维标签：
        [hpf1, hpf2, lpf1, lpf2, eq_low, eq_mid, eq_high, duration_ratio]
      - 使用 MERTEncoder 分别得到 A 段和 B 段的帧级 embedding
      - 计算每首歌的节拍相位位置编码 (sin, cos)
      - 作为输入：x1 = [H_A, PE_A], x2 = [H_B, PE_B]
        其中 H_* 维度 = d_mert, PE_* 维度 = 2，时间维均为 T
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

        audio_A_seg = self._load_last_n_seconds(audio_A_full, seg_seconds)
        audio_B_seg = self._load_first_n_seconds(audio_B_full, seg_seconds)

        # 4) 提取 MERT 帧特征并重采样到 T 帧
        H_A = self.encoder.extract_embeddings(audio_A_seg)  # (Ta, d)
        H_B = self.encoder.extract_embeddings(audio_B_seg)  # (Tb, d)
        H_A = self.encoder.resample_frames(H_A, self.target_frames)  # (T, d)
        H_B = self.encoder.resample_frames(H_B, self.target_frames)  # (T, d)

        # 5) 计算每首歌的节拍相位编码 (T, 2)
        PE_A = bpm_position_encoding(path_A, target_frames=self.target_frames, sr=self.encoder.target_sr)
        PE_B = bpm_position_encoding(path_B, target_frames=self.target_frames, sr=self.encoder.target_sr)

        # 6) 组装输入 (T, d+2)
        X1 = np.concatenate([H_A, PE_A], axis=-1).astype(np.float32)
        X2 = np.concatenate([H_B, PE_B], axis=-1).astype(np.float32)

        # 7) 复制标签到每一帧 (T, 8)
        Y = np.repeat(y_vec[None, :], self.target_frames, axis=0)  # (T, 8)

        X1_t = torch.from_numpy(X1)  # (T, d+2)
        X2_t = torch.from_numpy(X2)  # (T, d+2)
        Y_t = torch.from_numpy(Y)    # (T, 8)

        return X1_t, X2_t, Y_t, {
            "path_A": path_A,
            "path_B": path_B,
            "duration_ratio": duration_ratio
        }


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: str) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for X1, X2, Y, _meta in loader:
        X1 = X1.to(device)      # (B, T, d+2)
        X2 = X2.to(device)      # (B, T, d+2)
        Y = Y.to(device)        # (B, T, 8)

        optimizer.zero_grad(set_to_none=True)
        Y_pred, _ = model(X1, X2)     # (B, T, 8)
        loss = F.mse_loss(Y_pred, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = X1.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: str) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    for X1, X2, Y, _meta in loader:
        X1 = X1.to(device)
        X2 = X2.to(device)
        Y = Y.to(device)
        Y_pred, _ = model(X1, X2)
        loss = F.mse_loss(Y_pred, Y)

        bs = X1.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / max(1, total_count)


def main():
    # 配置
    train_tsv = os.path.join(_ROOT_DIR, "data", "train.tsv")
    valid_tsv = os.path.join(_ROOT_DIR, "data", "valid.tsv")
    # 可选：如果 manifest 的根目录不是你本地的真实路径，使用 override_root
    override_root = os.path.join(_ROOT_DIR, "data", "wav_dir")

    target_frames = 200
    max_transition_seconds = 20
    batch_size = 1            # MERT + 推理较重，建议从 1 开始
    num_workers = 0
    epochs = 3
    lr = 2e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 数据对
    train_pairs = build_pairs_from_manifest(train_tsv, override_root=override_root)
    valid_pairs = build_pairs_from_manifest(valid_tsv, override_root=override_root)

    print(f"[Data] #train pairs={len(train_pairs)}, #valid pairs={len(valid_pairs)}")

    # 编码器与模型
    mert = MERTEncoder(model_name="m-a-p/MERT-v1-95M")
    d_mert = int(mert.embedding_dim)
    in_dim_total = 2 * (d_mert + 2)   # [H_A, PE_A] + [H_B, PE_B]

    model = MixerTransformer(
        in_dim=in_dim_total,
        d_model=512,
        nhead=8,
        num_layers=4,
        dsp_dim=8
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
        X1_list, X2_list, Y_list, M_list = [], [], [], []
        for X1, X2, Y, M in batch:
            X1_list.append(X1)
            X2_list.append(X2)
            Y_list.append(Y)
            M_list.append(M)
        X1_t = torch.stack(X1_list, dim=0)
        X2_t = torch.stack(X2_list, dim=0)
        Y_t = torch.stack(Y_list, dim=0)
        return X1_t, X2_t, Y_t, M_list

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
    ckpt_path = os.path.join(_ROOT_DIR, "models", "mixer_checkpoint.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        valid_loss = evaluate(model, valid_loader, device)
        t1 = time.time()
        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.6f} "
              f"valid_loss={valid_loss:.6f} time={t1 - t0:.1f}s")

        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "in_dim": in_dim_total,
                    "d_model": 512,
                    "nhead": 8,
                    "num_layers": 4,
                    "dsp_dim": 8,
                    "target_frames": target_frames,
                }
            }, ckpt_path)
            print(f"[Checkpoint] Saved to: {ckpt_path}")


if __name__ == "__main__":
    main()


