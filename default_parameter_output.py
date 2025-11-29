import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt


################################################################################
# 0. Compute beat position (absolute beats)
################################################################################
def estimate_bpm(audio, sr,
                 min_bpm=70.0,
                 max_bpm=160.0,
                 default_bpm=120.0):
    """
    更鲁棒的 BPM 估计：
      1) 用 onset envelope + librosa.beat.tempo 估计 tempo
      2) 纠正极慢/极快为 half/double-time
      3) clamp 到 [min_bpm, max_bpm]
    """

    # 1) 计算 onset envelope（比直接 beat_track 更稳）
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)

    # 2) 用 tempo() 估一个全局 BPM（用 median 更稳）
    try:
        tempo = librosa.beat.tempo(
            onset_envelope=onset_env,
            sr=sr,
            aggregate=np.median,   # 比 mean 稳定
            max_tempo=320.0
        )[0]
    except Exception as e:
        print(f"[WARN] tempo() failed: {e}")
        return default_bpm

    if tempo <= 0 or np.isnan(tempo):
        print("[WARN] tempo invalid, fallback default bpm")
        return default_bpm

    # 3) half-time / double-time 纠正
    # 如果太快，就往下折半；如果太慢，就往上翻倍
    while tempo > max_bpm:
        tempo /= 2.0
    while tempo < min_bpm:
        tempo *= 2.0

    # 4) 最终 clamp
    tempo = float(np.clip(tempo, min_bpm, max_bpm))

    return tempo

def compute_beat_position(audio, sr, T=200):

    bpm = estimate_bpm(audio, sr)
    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, T)

    beat_pos = (bpm / 60.0) * time_axis
    return beat_pos, bpm, time_axis


################################################################################
# Duration ratio based on BPM difference
################################################################################
def compute_duration_ratio(bpmA, bpmB,
                           min_ratio=0.15,
                           max_ratio=1.0):

    diff = abs(bpmA - bpmB)

    if diff <= 8:
        ratio = max_ratio
    elif diff <= 15:
        ratio = max_ratio - (diff - 8) * (0.5 / 5)
    elif diff <= 30:
        ratio = 0.5 - (diff - 15) * (0.3 / 7)
    else:
        ratio = min_ratio

    return float(np.clip(ratio, min_ratio, max_ratio))


################################################################################
# BPM Morphing scheduler
################################################################################
def bpm_morph_scheduler(T, beat_pos_A, beat_pos_B):
    t = np.linspace(0, 1, T)
    return (1 - t) * beat_pos_A + t * beat_pos_B


################################################################################
# DSP schedule
################################################################################
def bpm_scheduler_dsp(T, beat_pos):

    t = np.linspace(0, 1, T)

    gainA = (1 - t) + 0.05 * np.sin(2 * np.pi * beat_pos)
    gainB = t + 0.05 * np.sin(2 * np.pi * beat_pos)

    hpf1 = 60       # 0–50%  (almost no cut)
    hpf2 = 300      # 50–100% (more cut)

    lpf1 = 14000    # 0–50%  (full brightness)
    lpf2 = 5000     # 50–100% (darken incoming track)

    # Stage mask: 前半段=0, 后半段=1
    stage = (t >= 0.5).astype(float)

    # Interpolation
    hpf = hpf1 * (1 - stage) + hpf2 * stage
    lpf = lpf1 * (1 - stage) + lpf2 * stage

    eq_low = np.ones(T)
    eq_mid = np.ones(T)
    eq_high = np.ones(T)

    loud = np.concatenate([
        np.linspace(1, 0.7, T//2),
        np.linspace(0.7, 1, T - T//2)
    ])
    loud *= (1 - 0.25 * np.exp(-2 * np.mod(beat_pos, 1)))

    return np.stack([gainA, gainB, hpf, lpf, eq_low, eq_mid, eq_high, loud], axis=-1)


################################################################################
# 3. Visualization
################################################################################
def plot_dsp(dsp_params, beat_pos, save_path="dsp_plot.png"):

    T = dsp_params.shape[0]
    t = np.linspace(0, 1, T)

    gainA, gainB = dsp_params[:, 0], dsp_params[:, 1]
    hpf, lpf = dsp_params[:, 2], dsp_params[:, 3]
    eq_low, eq_mid, eq_high = dsp_params[:, 4], dsp_params[:, 5], dsp_params[:, 6]
    loud = dsp_params[:, 7]

    plt.figure(figsize=(16, 22))

    plt.subplot(6, 1, 1); plt.title("Gain")
    plt.plot(t, gainA); plt.plot(t, gainB); plt.grid()

    plt.subplot(6, 1, 2); plt.title("Filters")
    plt.plot(t, hpf); plt.plot(t, lpf); plt.grid()

    plt.subplot(6, 1, 3); plt.title("EQ")
    plt.plot(t, eq_low); plt.plot(t, eq_mid); plt.plot(t, eq_high); plt.grid()

    plt.subplot(6, 1, 4); plt.title("Loudness")
    plt.plot(t, loud); plt.grid()

    plt.subplot(6, 1, 5); plt.title("Beat Position")
    plt.plot(t, beat_pos); plt.grid()

    plt.subplot(6, 1, 6); plt.title("Beat Phase")
    plt.plot(t, np.mod(beat_pos, 1)); plt.grid()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Saved DSP plot] {save_path}")
    plt.close()


################################################################################
# 4. DSP Engine
################################################################################
class DSPEngine:
    def __init__(self, sr=16000):
        self.sr = sr

    def _filter_freq(self, audio, cutoff, mode="lowpass"):
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)

        if mode == "lowpass":
            fft[freqs > cutoff] = 0
        else:
            fft[freqs < cutoff] = 0

        return np.fft.irfft(fft, n=len(audio))

    def _apply_eq(self, audio,  elo, emid, ehi):

        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)
        fft[freqs < 200] *= elo
        fft[(freqs >= 200) & (freqs < 2000)] *= emid
        fft[freqs >= 2000] *= ehi

        fft[(freqs >= 200) & (freqs < 400)] *= 1.8
        fft[(freqs >= 600) & (freqs < 1200)] *= 1.4
        fft[(freqs >= 2000) & (freqs < 4000)] *= 0.55
        fft[(freqs >= 5000) & (freqs < 8000)] *= 0.70
        fft[freqs >= 10000] *= 0.85

        return np.fft.irfft(fft, n=len(audio))

    def render_transition(self, A, B, dsp_params):

        T = dsp_params.shape[0]
        L = min(len(A), len(B))

        x = np.linspace(0, 1, T)
        xf = np.linspace(0, 1, L)
        interp = lambda p: np.interp(xf, x, p)

        gainA = interp(dsp_params[:, 0])
        gainB = interp(dsp_params[:, 1])
        hpf_cut = np.mean(dsp_params[:, 2])
        lpf_cut = np.mean(dsp_params[:, 3])
        eq_low  = interp(dsp_params[:, 4])
        eq_mid  = interp(dsp_params[:, 5])
        eq_high = interp(dsp_params[:, 6])
        loud = interp(dsp_params[:, 7])

        # Filters
        A_f = self._filter_freq(A[:L], hpf_cut, mode="highpass")
        B_f = self._filter_freq(B[:L], lpf_cut, mode="lowpass")
        A_p = self._apply_eq(A_f, np.mean(eq_low), np.mean(eq_mid), np.mean(eq_high))
        B_p = self._apply_eq(B_f, np.mean(eq_low), np.mean(eq_mid), np.mean(eq_high))


        mix = gainA * A_p + gainB * B_p
        mix *= loud

        return mix / (np.max(np.abs(mix)) + 1e-9) * 0.95


################################################################################
# 5. Create ONLY transition (AUTOMATIC duration_ratio)
################################################################################
def make_transition(
    audioA_path,
    audioB_path,
    out_path="transition.wav",
    plot_path="dsp_plot.png",
    sr=16000,
    max_transition_seconds=20
):

    # Load songs
    A, sra = sf.read(audioA_path)
    B, srb = sf.read(audioB_path)

    if A.ndim > 1: A = A.mean(axis=1)
    if B.ndim > 1: B = B.mean(axis=1)

    A = librosa.resample(A, orig_sr=sra, target_sr=sr)
    B = librosa.resample(B, orig_sr=srb, target_sr=sr)

    # Compute BPM first (short windows)
    beat_A, bpmA, _ = compute_beat_position(A[-5*sr:], sr, 200)
    beat_B, bpmB, _ = compute_beat_position(B[:5*sr], sr, 200)

    # Auto duration ratio
    duration_ratio = compute_duration_ratio(bpmA, bpmB)
    print(f"[Transition] BPM_A={float(bpmA):.2f}, BPM_B={float(bpmB):.2f}, duration_ratio={duration_ratio:.3f}")

    N = int(duration_ratio * max_transition_seconds * sr)

    A_cut = A[-N:]
    B_cut = B[:N]

    beat_A, _, _ = compute_beat_position(A_cut, sr, 200)
    beat_B, _, _ = compute_beat_position(B_cut, sr, 200)

    beat_mix = bpm_morph_scheduler(200, beat_A, beat_B)
    dsp_params = bpm_scheduler_dsp(200, beat_mix)
    plot_dsp(dsp_params, beat_mix, plot_path)

    engine = DSPEngine(sr)
    transition = engine.render_transition(A_cut, B_cut, dsp_params)

    sf.write(out_path, transition, sr)
    print(f"[DONE] Transition saved → {out_path}")

    return transition



################################################################################
# 6. Full Song (A + transition + rest of B) — AUTO DURATION RATIO
################################################################################
def make_full_song(
    audioA_path,
    audioB_path,
    out_path="full_mix.wav",
    sr=16000,
    max_transition_seconds=20
):

    A, sra = sf.read(audioA_path)
    B, srb = sf.read(audioB_path)

    # mono
    if A.ndim > 1: A = A.mean(axis=1)
    if B.ndim > 1: B = B.mean(axis=1)

    # resample
    A = librosa.resample(A, orig_sr=sra, target_sr=sr)
    B = librosa.resample(B, orig_sr=srb, target_sr=sr)

    # detect BPM first
    beat_A, bpmA, _ = compute_beat_position(A[-5*sr:], sr, 200)
    beat_B, bpmB, _ = compute_beat_position(B[:5*sr], sr, 200)

    # auto duration ratio
    duration_ratio = compute_duration_ratio(bpmA, bpmB)
    print(f"[Full Song] BPM_A={float(bpmA):.2f}, BPM_B={float(bpmB):.2f}, duration_ratio={duration_ratio:.3f}")

    N = int(duration_ratio * max_transition_seconds * sr)

    A_cut = A[-N:]
    B_cut = B[:N]
    B_rest = B[N:]

    # BPM-driven DSP
    beat_A, _, _ = compute_beat_position(A_cut, sr, 200)
    beat_B, _, _ = compute_beat_position(B_cut, sr, 200)

    beat_mix = bpm_morph_scheduler(200, beat_A, beat_B)
    dsp_params = bpm_scheduler_dsp(200, beat_mix)

    engine = DSPEngine(sr)
    transition = engine.render_transition(A_cut, B_cut, dsp_params)

    full_mix = np.concatenate([A[:-N], transition, B_rest])
    full_mix = full_mix / (np.max(np.abs(full_mix)) + 1e-9) * 0.98

    sf.write(out_path, full_mix, sr)
    print(f"[DONE] Full song saved → {out_path}")

    return full_mix



################################################################################
# Main Example
################################################################################
if __name__ == "__main__":

    make_transition(
        "songA.mp3",
        "songB.mp3",
        out_path="transition.wav",
        plot_path="dsp_plot.png",
    )

    make_full_song(
        "songA.mp3",
        "songB.mp3",
        out_path="full_mix.wav",
    )
