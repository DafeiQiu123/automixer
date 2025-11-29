import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt


################################################################################
# 0. Compute beat position (absolute beats) for one audio segment
################################################################################
def compute_beat_position(audio, sr, T=200):
    """
    输出：
        beat_pos[t] = (bpm / 60) * time_seconds
        bpm
    """

    # 1) librosa beat detection
    bpm, _ = librosa.beat.beat_track(y=audio, sr=sr)
    if bpm < 1:
        print("WARNING: BPM detection failed → fallback BPM=120.")
        bpm = 120.0

    # 2) absolute beat position
    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, T)
    beat_pos = (bpm / 60.0) * time_axis

    return beat_pos, bpm, time_axis


################################################################################
# 1. Beat Morphing Scheduler: combine A's BPM and B's BPM
################################################################################
def bpm_morph_scheduler(T, beat_pos_A, beat_pos_B):
    """
    最终 beat_pos(t) 同时反映 A & B 的节奏：
        前半段跟 A
        后半段跟 B
        中间平滑 morph
    """

    t = np.linspace(0, 1, T)
    beat_pos_mix = (1 - t) * beat_pos_A + t * beat_pos_B

    return beat_pos_mix


################################################################################
# 2. DSP parameters driven by beat_pos_mix
################################################################################
def bpm_scheduler_dsp(T, beat_pos):

    t = np.linspace(0, 1, T)

    # Gain with beat modulation
    gainA = (1 - t) + 0.05 * np.sin(2*np.pi*beat_pos)
    gainB = t         + 0.05 * np.sin(2*np.pi*beat_pos)

    # Filters
    filter_mask = np.clip((t - 0.3) / 0.4, 0, 1)

    hpf = 60 + 1500 * filter_mask
    lpf = 12000 - 4000 * filter_mask

    # EQ
    eq_low  = 1 - 0.1 * (np.sin(2*np.pi*beat_pos) > 0.7)
    eq_mid  = 1 + 0.1 * np.sin(4*np.pi*beat_pos)
    eq_high = np.ones(T)

    # Loudness
    min_loud = 0.7
    half = T // 2
    loud_down = np.linspace(1, min_loud, half)
    loud_up   = np.linspace(min_loud, 1, T-half)
    loud = np.concatenate([loud_down, loud_up])

    loud *= (1 - 0.25*np.exp(-2 * np.mod(beat_pos, 1)))

    return np.stack(
        [gainA, gainB, hpf, lpf, eq_low, eq_mid, eq_high, loud],
        axis=-1
    )


################################################################################
# 3. Visualization
################################################################################
def plot_dsp(dsp_params, beat_pos, save_path="dsp_plot.png"):

    T = dsp_params.shape[0]
    t = np.linspace(0,1,T)

    gainA, gainB = dsp_params[:,0], dsp_params[:,1]
    hpf,   lpf   = dsp_params[:,2], dsp_params[:,3]
    eq_low, eq_mid, eq_high = dsp_params[:,4], dsp_params[:,5], dsp_params[:,6]
    loud = dsp_params[:,7]

    plt.figure(figsize=(16, 22))

    plt.subplot(6,1,1); plt.title("Gain Curves")
    plt.plot(t, gainA); plt.plot(t, gainB); plt.grid()

    plt.subplot(6,1,2); plt.title("Filters")
    plt.plot(t, hpf); plt.plot(t, lpf); plt.grid()

    plt.subplot(6,1,3); plt.title("EQ Curves")
    plt.plot(t, eq_low); plt.plot(t, eq_mid); plt.plot(t, eq_high); plt.grid()

    plt.subplot(6,1,4); plt.title("Loudness")
    plt.plot(t, loud); plt.grid()

    plt.subplot(6,1,5); plt.title("Absolute Beat Position (Morphed)")
    plt.plot(t, beat_pos); plt.grid()

    plt.subplot(6,1,6); plt.title("Beat Phase = beat_pos % 1")
    plt.plot(t, np.mod(beat_pos, 1)); plt.grid()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Saved DSP Plot] {save_path}")
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
        if mode=="lowpass": fft[freqs > cutoff] = 0
        else:               fft[freqs < cutoff] = 0
        return np.fft.irfft(fft, n=len(audio))

    def _apply_eq(self, audio, elo, emid, ehi):
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)
        fft[freqs<200] *= elo
        fft[(freqs>=200)&(freqs<2000)] *= emid
        fft[freqs>=2000] *= ehi
        return np.fft.irfft(fft, n=len(audio))

    def render_transition(self, A, B, dsp_params):

        T = dsp_params.shape[0]
        length = min(len(A), len(B))

        x = np.linspace(0,1,T)
        x_full = np.linspace(0,1,length)
        interp = lambda p: np.interp(x_full, x, p)

        gainA = interp(dsp_params[:,0])
        gainB = interp(dsp_params[:,1])
        hpf_cut = np.mean(dsp_params[:,2])
        lpf_cut = np.mean(dsp_params[:,3])
        eq_low  = np.mean(dsp_params[:,4])
        eq_mid  = np.mean(dsp_params[:,5])
        eq_high = np.mean(dsp_params[:,6])
        loud    = interp(dsp_params[:,7])

        if hpf_cut < 100:         
            A_processed = A[:length]   
        else:
            A_f = self._filter_freq(A[:length], hpf_cut, mode="highpass")
            A_processed = self._apply_eq(A_f, eq_low, eq_mid, eq_high)

        if lpf_cut > 10000:       
            B_processed = B[:length]  
        else:
            B_f = self._filter_freq(B[:length], lpf_cut, mode="lowpass")
            B_processed = self._apply_eq(B_f, eq_low, eq_mid, eq_high)

        mix = gainA * A_processed + gainB * B_processed
        mix *= loud

        mix /= np.max(np.abs(mix)) + 1e-9
        mix *= 0.95
        return mix


################################################################################
# 5. MAIN
################################################################################
def make_transition(
    audioA_path,
    audioB_path,
    out_path="transition.wav",
    plot_path="dsp_plot.png",
    sr=16000,
    duration_ratio=0.5,
    max_transition_seconds=20
):

    A, sra = sf.read(audioA_path)
    B, srb = sf.read(audioB_path)

    if A.ndim>1: A=A.mean(axis=1)
    if B.ndim>1: B=B.mean(axis=1)

    A = librosa.resample(y=A, orig_sr=sra, target_sr=sr)
    B = librosa.resample(y=B, orig_sr=srb, target_sr=sr)

    N = int(duration_ratio * max_transition_seconds * sr)
    A_cut = A[-N:]
    B_cut = B[:N]

    # Compute beat positions for both songs
    beat_A, bpm_A, _ = compute_beat_position(A_cut, sr, T=200)
    beat_B, bpm_B, _ = compute_beat_position(B_cut, sr, T=200)

    # Smooth morphing
    beat_mix = bpm_morph_scheduler(T=200, beat_pos_A=beat_A, beat_pos_B=beat_B)

    dsp_params = bpm_scheduler_dsp(T=200, beat_pos=beat_mix)

    plot_dsp(dsp_params, beat_mix, save_path=plot_path)

    engine = DSPEngine(sr)
    audio_out = engine.render_transition(A_cut, B_cut, dsp_params)
    sf.write(out_path, audio_out, sr)

    print(f"[DONE] Transition saved to {out_path}")
    print(f"BPM_A = {bpm_A}, BPM_B = {bpm_B}")


################################################################################
# Run Example
################################################################################
if __name__ == "__main__":
    make_transition(
        "songA.mp3",
        "songB.mp3",
        out_path="transition.wav",
        plot_path="dsp_plot.png",
        duration_ratio=1.0,
    )
