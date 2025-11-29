import numpy as np
import librosa
import soundfile as sf


###############################################################################
# 0. 提取 beat_phase (librosa)
###############################################################################
def get_beat_phase(audio, sr, T=200):
    """
    使用 librosa 自动检测拍点并生成 beat phase 序列。
    beat_phase(t) ∈ [0,1)
    """
    try:
        tempo, beat_times = librosa.beat.beat_track(y=audio, sr=sr, units="time")
    except:
        print("WARNING: librosa beat tracking failed → using linear phase.")
        return np.linspace(0, 1, T)

    if len(beat_times) < 2:
        print("WARNING: insufficient beat detections → using linear phase.")
        return np.linspace(0, 1, T)

    duration = len(audio) / sr
    frame_times = np.linspace(0, duration, T)

    beat_phase = np.zeros(T)
    idx = 0
    for i, t in enumerate(frame_times):
        while idx + 1 < len(beat_times) and beat_times[idx+1] < t:
            idx += 1

        beat_len = beat_times[idx+1] - beat_times[idx] if idx+1 < len(beat_times) else 1e-8
        beat_phase[i] = (t - beat_times[idx]) / beat_len

    return beat_phase % 1.0


###############################################################################
# 1. BPM-aware DSP template (EQ + loudness)
###############################################################################
def bpm_dsp_auto(T, beat_phase):
    """
    自动生成 EQ 与 loudness，不需要模型预测。
    """

    t = np.linspace(0, 1, T)

    # --- EQ ---
    eq_low = 1 - 0.15 * (beat_phase > 0.9)
    eq_mid = 1 + 0.1 * np.sin(2*np.pi*beat_phase)
    eq_high = np.ones(T)

    # --- Loudness linear down-up + beat pumping ---
    min_loud = 0.7
    half = T // 2
    loud_down = np.linspace(1, min_loud, half)
    loud_up   = np.linspace(min_loud, 1, T - half)
    loud = np.concatenate([loud_down, loud_up])

    # add pumping
    loud *= (1 - 0.25 * np.exp(-4 * beat_phase))

    return eq_low, eq_mid, eq_high, loud


###############################################################################
# 2. DSP ENGINE
###############################################################################
class DSPEngine:
    def __init__(self, sr=16000):
        self.sr = sr

    # -------------------------
    def _filter_freq(self, audio, cutoff, mode="lowpass"):
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)

        if mode == "lowpass":
            fft[freqs > cutoff] = 0
        else:
            fft[freqs < cutoff] = 0

        return np.fft.irfft(fft, n=len(audio))

    # -------------------------
    def _apply_eq(self, audio, elo, emid, ehi):
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)

        fft[freqs < 200] *= elo
        fft[(freqs>=200)&(freqs<2000)] *= emid
        fft[freqs>=2000] *= ehi

        return np.fft.irfft(fft, n=len(audio))

    # -------------------------
    def render(self, A, B, gainA, gainB, hpf, lpf, eq_low, eq_mid, eq_high, loud):
        L = min(len(A), len(B))

        # Filter
        A_f = self._filter_freq(A[:L], np.mean(hpf), mode="highpass")
        B_f = self._filter_freq(B[:L], np.mean(lpf), mode="lowpass")

        # EQ
        A_eq = self._apply_eq(A_f, np.mean(eq_low), np.mean(eq_mid), np.mean(eq_high))
        B_eq = self._apply_eq(B_f, np.mean(eq_low), np.mean(eq_mid), np.mean(eq_high))

        # Mix
        mix = gainA * A_eq + gainB * B_eq
        mix *= loud
        return mix


###############################################################################
# 3. 主函数：模型参数 + librosa beat → 最终音频
###############################################################################
def render_transition(
    audioA_path,
    audioB_path,
    dsp_seq,         # (T, 4): gainA, gainB, hpf, lpf
    duration_ratio,  # scalar
    out_path="transition.wav",
    sr=16000,
    max_transition_seconds=20
):
    T = dsp_seq.shape[0]

    # -----------------------------
    # Load audio
    # -----------------------------
    A_raw, sra = sf.read(audioA_path)
    B_raw, srb = sf.read(audioB_path)

    if A_raw.ndim > 1: A_raw = A_raw.mean(axis=1)
    if B_raw.ndim > 1: B_raw = B_raw.mean(axis=1)

    A = librosa.resample(A_raw, orig_sr=sra, target_sr=sr)
    B = librosa.resample(B_raw, orig_sr=srb, target_sr=sr)

    # -----------------------------
    # Duration
    # -----------------------------
    duration_sec = duration_ratio * max_transition_seconds
    N = int(duration_sec * sr)

    A_cut = A[-N:]
    B_cut = B[:N]

    # -----------------------------
    # Get beat-phase
    # -----------------------------
    beat_phase = get_beat_phase(A_cut, sr, T)

    # -----------------------------
    # Auto DSP from BPM (EQ, loud)
    # -----------------------------
    eq_low, eq_mid, eq_high, loud = bpm_dsp_auto(T, beat_phase)

    # -----------------------------
    # Extract model params
    # -----------------------------
    gainA = dsp_seq[:,0]
    gainB = dsp_seq[:,1]
    hpf   = dsp_seq[:,2]
    lpf   = dsp_seq[:,3]

    # Upsampling
    L = len(A_cut)
    x = np.linspace(0,1,T)
    x_full = np.linspace(0,1,L)

    gainA = np.interp(x_full, x, gainA)
    gainB = np.interp(x_full, x, gainB)
    loud  = np.interp(x_full, x, loud)

    # -----------------------------
    # Render DSP
    # -----------------------------
    engine = DSPEngine(sr)
    audio_out = engine.render(A_cut, B_cut, gainA, gainB,
                              hpf, lpf,
                              eq_low, eq_mid, eq_high,
                              loud)

    sf.write(out_path, audio_out, sr)
    print(f"[saved] {out_path}")
    return audio_out


###############################################################################
# Example
###############################################################################
if __name__ == "__main__":

    # Fake model output example
    T = 200
    dsp_seq = np.random.rand(T, 4)       # 模型输出
    duration_ratio = 0.6                 # 模型输出

    render_transition(
        "songA.mp3",
        "songB.mp3",
        dsp_seq,
        duration_ratio,
        out_path="transition.wav"
    )
