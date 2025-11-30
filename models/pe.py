import torch
import math


class DualBPMPositionalEncoding(torch.nn.Module):
    def __init__(self, hidden_dim, sampling_rate=24000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sampling_rate = sampling_rate

    def forward(self, x, time_a, bpm_a, bpm_b):
        """
        x: [1, T, D]
        time_a: int, number of frames in segment A
        bpm_a, bpm_b: scalar BPM for each segment
        """
        B, T, D = x.shape
        assert B == 1, "Batch size must be 1."

        time_a = int(time_a)
        time_b = T - time_a

        # Time index [0 ... T-1]
        t = torch.arange(T, dtype=torch.float32, device=x.device)

        # Seconds per beat
        period_a = 60.0 / bpm_a
        period_b = 60.0 / bpm_b

        # Frame index → beat index  (pos = beat number, not raw time)
        pos = torch.zeros_like(t)

        # A segment pos = t / (period * sr)
        pos[:time_a] = t[:time_a] / (period_a * self.sampling_rate)

        # B segment pos starts from zero: (t - time_a) / period_b
        pos[time_a:] = (t[time_a:] - time_a) / (period_b * self.sampling_rate)

        # Build PE
        pe = torch.zeros(T, D, device=x.device)
        half = D // 2

        # base frequencies
        freq = torch.exp(torch.arange(0, half, 2, device=x.device) * (-math.log(10000.0) / half))
        # relative scaling so model can feel bpm relationship
        freq_b = freq * (bpm_b / bpm_a)

        # PE_A for first half channels
        pe_a = torch.zeros(T, half, device=x.device)
        pe_a[:, 0::2] = torch.sin(pos.unsqueeze(-1) * freq)
        pe_a[:, 1::2] = torch.cos(pos.unsqueeze(-1) * freq)

        # PE_B for second half channels
        pe_b = torch.zeros(T, half, device=x.device)
        pe_b[:, 0::2] = torch.sin(pos.unsqueeze(-1) * freq_b)
        pe_b[:, 1::2] = torch.cos(pos.unsqueeze(-1) * freq_b)

        # Combine
        pe[:, :half] = pe_a
        pe[:, half:] = pe_b

        return x + pe.unsqueeze(0)

if __name__ == "__main__":
    pe = DualBPMPositionalEncoding(hidden_dim=128)
    x = torch.randn(1, 400, 768)
    time_a = 100
    bpm_a = 120
    bpm_b = 130
    pe_x = pe(x, time_a, bpm_a, bpm_b)
    print(pe_x.shape)