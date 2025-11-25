# data/dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset

from models.mert_encoder import MERTEncoder
from models.bpm_encoding import bpm_position_encoding
from dsp.dsp_rules import rule_based_dsp


class ABTransitionDataset(Dataset):
    """
    Dataset that:
      - takes song pairs (A, B)
      - extracts MERT embeddings for outro of A and intro of B
      - adds BPM position encoding
      - generates rule-based DSP labels
    """

    def __init__(self,
                 pairs: list[tuple[str, str]],
                 mert_encoder: MERTEncoder,
                 target_frames: int = 200,
                 outro_sec: float = 10.0,
                 intro_sec: float = 10.0):
        self.pairs = pairs
        self.encoder = mert_encoder
        self.target_frames = target_frames
        self.outro_sec = outro_sec
        self.intro_sec = intro_sec

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        A_path, B_path = self.pairs[idx]

        # For simplicity, we take last outro_sec of A and first intro_sec of B.
        # You might want to precompute song durations; here we just assume.
        # To avoid needing duration, just encode full and let MERTEncoder handle it.
        H_A = self.encoder.encode_segment(A_path, start_sec=None,
                                          end_sec=None,
                                          target_frames=self.target_frames)
        H_B = self.encoder.encode_segment(B_path, start_sec=None,
                                          end_sec=None,
                                          target_frames=self.target_frames)

        # BPM PE
        PE_A = bpm_position_encoding(A_path, target_frames=self.target_frames)
        PE_B = bpm_position_encoding(B_path, target_frames=self.target_frames)

        # Concatenate features per frame: [H_A, H_B, PE_A, PE_B]
        X = np.concatenate(
            [H_A, H_B, PE_A, PE_B],
            axis=-1
        )  # shape (T, 2*d_mert + 4)

        # Rule-based DSP teacher
        Y = rule_based_dsp(self.target_frames)  # (T, 8)

        X_t = torch.tensor(X, dtype=torch.float32)
        Y_t = torch.tensor(Y, dtype=torch.float32)

        return X_t, Y_t
