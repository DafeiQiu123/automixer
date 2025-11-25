# models/transformer_mixer.py
import torch
import torch.nn as nn


class MixerTransformer(nn.Module):
    """
    Transformer that maps per-frame music features -> DSP parameters.
    """

    def __init__(self,
                 in_dim: int,       # 2*d_mert + 4
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dsp_dim: int = 8):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.output_head = nn.Linear(d_model, dsp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, in_dim)
        returns: (B, T, dsp_dim)
        """
        h = self.input_proj(x)        # (B, T, d_model)
        h = self.transformer(h)       # (B, T, d_model)
        y = self.output_head(h)       # (B, T, dsp_dim)
        return y
