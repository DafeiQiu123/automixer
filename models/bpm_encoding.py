# models/bpm_encoding.py
import numpy as np
import librosa


def bpm_position_encoding(audio_path: str,
                          target_frames: int = 200,
                          sr: int = 16000) -> np.ndarray:
    """
    Compute a simple beat-phase based positional encoding.
    使用节拍相位生成位置编码，返回 (T, 2): [sin(phi), cos(phi)]
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    # Beat tracking: beat_frames are frame indices
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    if len(beat_frames) < 2:
        # Fallback: linear phase 0->1
        phase = np.linspace(0, 1, target_frames)
        return np.stack([np.sin(2 * np.pi * phase),
                         np.cos(2 * np.pi * phase)], axis=-1)

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # We create target_frames timestamps
    max_t = beat_times[-1]
    frame_times = np.linspace(0, max_t, target_frames)

    phase = np.zeros(target_frames)
    idx = 0
    for i, t in enumerate(frame_times):
        while idx + 1 < len(beat_times) and beat_times[idx + 1] < t:
            idx += 1
        t0 = beat_times[idx]
        t1 = beat_times[idx + 1] if idx + 1 < len(beat_times) else t0 + 1e-6
        phase[i] = (t - t0) / max(t1 - t0, 1e-6)  # in [0,1]

    pos_enc = np.stack(
        [np.sin(2 * np.pi * phase),
         np.cos(2 * np.pi * phase)],
        axis=-1
    )  # (T, 2)

    return pos_enc
