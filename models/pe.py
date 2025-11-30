import torch
import math
import matplotlib.pyplot as plt


class DualBPMPositionalEncoding(torch.nn.Module):
    def __init__(self, hidden_dim, sampling_rate=24000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sampling_rate = sampling_rate

    def forward(self, x, bpm_a, bpm_b):
        """
        x:     [B, T, D]
        bpm_a: [B]
        bpm_b: [B]
        """
        B, T, D = x.shape
        half_T = T // 2  # split directly in the middle

        # -----------------------------------------
        # 1) Time indices: [B, T]
        # -----------------------------------------
        t = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

        # -----------------------------------------
        # 2) Create masks for A / B
        # -----------------------------------------
        mask_A = t < half_T
        mask_B = ~mask_A

        # -----------------------------------------
        # 3) Compute beat periods
        # -----------------------------------------
        period_a = (60.0 / bpm_a).unsqueeze(1)  # [B,1]
        period_b = (60.0 / bpm_b).unsqueeze(1)  # [B,1]

        # -----------------------------------------
        # 4) Compute pos: beat index per segment
        # -----------------------------------------
        pos = torch.zeros(B, T, device=x.device)

        # A
        pos[mask_A] = t[mask_A] / (period_a.expand_as(t)[mask_A] * self.sampling_rate)

        # B
        shifted_t_B = t[mask_B] - half_T
        pos[mask_B] = shifted_t_B / (period_b.expand_as(t)[mask_B] * self.sampling_rate)

        # -----------------------------------------
        # 5) Sinusoidal PE (A/B half mixing)
        # -----------------------------------------
        half = D // 2

        freq = torch.exp(
            torch.arange(0, half, 2, device=x.device) * (-math.log(10000.0) / half)
        )
        freq = freq.unsqueeze(0).unsqueeze(0)  # [1,1,H/4]

        freq_b = freq * (bpm_b / bpm_a).unsqueeze(1).unsqueeze(2)

        pos_exp = pos.unsqueeze(-1)

        pe_a = torch.zeros(B, T, half, device=x.device)
        pe_a[:, :, 0::2] = torch.sin(pos_exp * freq)
        pe_a[:, :, 1::2] = torch.cos(pos_exp * freq)

        pe_b = torch.zeros(B, T, half, device=x.device)
        pe_b[:, :, 0::2] = torch.sin(pos_exp * freq_b)
        pe_b[:, :, 1::2] = torch.cos(pos_exp * freq_b)

        pe = torch.cat([pe_a, pe_b], dim=-1)

        return x + pe, pos   # also return pos for debugging/visualization


# -----------------------------------------
# Visualization helper
# -----------------------------------------
def visualize_pos(pos, bpm_a, bpm_b, sampling_rate=24000):
    """
    pos: [T]  -- one sample's beat-index curve
    """
    T = pos.shape[0]
    half_T = T // 2
    t = torch.arange(T)

    period_A = 60 / bpm_a
    period_B = 60 / bpm_b

    # Compute theoretical beat frames
    # -----------------------------
    beats_A = torch.arange(0, int(pos[:half_T].max().item()) + 2)
    beat_frames_A = (beats_A * period_A * sampling_rate).long()

    beats_B = torch.arange(0, int(pos[half_T:].max().item()) + 2)
    beat_frames_B = half_T + (beats_B * period_B * sampling_rate).long()

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(14, 5))
    plt.plot(t, pos, label="pos (beat index)", linewidth=2)

    # Draw vertical beat lines
    for bf in beat_frames_A:
        if bf < half_T:
            plt.axvline(bf, color='red', linestyle='--', alpha=0.4)

    for bf in beat_frames_B:
        if bf < T:
            plt.axvline(bf, color='green', linestyle='--', alpha=0.4)

    plt.title(f"Beat alignment visualization\nA BPM={bpm_a}, B BPM={bpm_b}")
    plt.xlabel("Frame index")
    plt.ylabel("Beat index")
    plt.grid(True)
    plt.legend()
    plt.show()


# -----------------------------------------
# MAIN
# -----------------------------------------
if __name__ == "__main__":
    B = 4      # batch size
    T = 8000   # frames
    D = 512    # dim

    x = torch.zeros(B, T, D)

    # Random BPM for testing
    bpm_a = torch.tensor([120, 90, 150, 110], dtype=torch.float32)
    bpm_b = torch.tensor([100, 140, 60, 130], dtype=torch.float32)

    pe_layer = DualBPMPositionalEncoding(D)
    out, pos_batch = pe_layer(x, bpm_a, bpm_b)

    print("Output shape:", out.shape)
    print("Pos shape:", pos_batch.shape)

    # Visualize sample 0
    print("\nVisualizing sample 0...")
    visualize_pos(pos_batch[0], bpm_a[0], bpm_b[0])