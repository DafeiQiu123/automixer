# inference.py
import torch
import torchaudio
import numpy as np

from models.mert_encoder import MERTEncoder
from models.bpm_encoding import bpm_position_encoding
from models.transformer_mixer import MixerTransformer
from dsp.dsp_engine import DSPEngine


def load_mono(path: str, sr: int = 16000) -> np.ndarray:
    audio, ori_sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if ori_sr != sr:
        audio = torchaudio.functional.resample(audio, ori_sr, sr)
    return audio.squeeze(0).numpy()


def infer_transition(A_path: str, B_path: str,
                     model_ckpt: str = "bpm_mixer.pt",
                     target_frames: int = 200,
                     sr: int = 16000,
                     out_path: str = "transition.wav"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load encoder & model
    mert = MERTEncoder()
    d_mert = mert.embedding_dim
    in_dim = 2 * d_mert + 4

    model = MixerTransformer(
        in_dim=in_dim,
        d_model=512,
        nhead=8,
        num_layers=4,
        dsp_dim=8
    )
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.to(device)
    model.eval()

    # 2) MERT embeddings
    H_A = mert.encode_segment(A_path, start_sec=None, end_sec=None,
                              target_frames=target_frames)
    H_B = mert.encode_segment(B_path, start_sec=None, end_sec=None,
                              target_frames=target_frames)

    # 3) BPM PE
    PE_A = bpm_position_encoding(A_path, target_frames=target_frames)
    PE_B = bpm_position_encoding(B_path, target_frames=target_frames)

    X = np.concatenate([H_A, H_B, PE_A, PE_B], axis=-1)  # (T, in_dim)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, in_dim)

    with torch.no_grad():
        dsp_pred = model(X_t).squeeze(0).cpu().numpy()  # (T, 8)

    # 4) Load raw audio for rendering
    audio_A = load_mono(A_path, sr=sr)
    audio_B = load_mono(B_path, sr=sr)

    # 5) DSP render
    eng = DSPEngine(sr=sr)
    transition = eng.render_transition(audio_A, audio_B, dsp_pred)
    eng.save_wav(out_path, transition)
    print(f"Saved transition to {out_path}")


if __name__ == "__main__":
    infer_transition("songA.wav", "songB.wav")
