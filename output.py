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

    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)

    try:
        tempo = librosa.beat.tempo(
            onset_envelope=onset_env,
            sr=sr,
            aggregate=np.median,
            max_tempo=320.0
        )[0]
    except:
        return default_bpm

    if tempo <= 0 or np.isnan(tempo):
        return default_bpm

    while tempo > max_bpm:
        tempo /= 2.0
    while tempo < min_bpm:
        tempo *= 2.0

    return float(np.clip(tempo, min_bpm, max_bpm))


def compute_beat_position(audio, sr, T=200):

    bpm = estimate_bpm(audio, sr)
    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, T)

    beat_pos = (bpm / 60.0) * time_axis
    return beat_pos, bpm, time_axis


def bpm_morph_scheduler(T, beat_pos_A, beat_pos_B):
    """
    Crossfade A->B beat positions smoothly
    """
    t = np.linspace(0, 1, T)
    return (1 - t) * beat_pos_A + t * beat_pos_B

################################################################################
# 1. DSP schedule — use ONLY model outputs
################################################################################
def model_dsp_schedule(T, beat_pos, params):
    """
    params:
        hpf1, hpf2,
        lpf1, lpf2,
        eq_low, eq_mid, eq_high,
        duration_ratio (not used here)
    """

    t = np.linspace(0, 1, T)

    # Gain — still beat-driven
    gainA = (1 - t) + 0.05 * np.sin(2 * np.pi * beat_pos)
    gainB = t         + 0.05 * np.sin(2 * np.pi * beat_pos)

    # 4 filter values from model
    hpf1 = params["hpf1"]
    hpf2 = params["hpf2"]
    lpf1 = params["lpf1"]
    lpf2 = params["lpf2"]

    stage = (t >= 0.5).astype(float)
    hpf = hpf1 * (1 - stage) + hpf2 * stage
    lpf = lpf1 * (1 - stage) + lpf2 * stage

    # EQ from model
    eq_low  = np.ones(T) * params["eq_low"]
    eq_mid  = np.ones(T) * params["eq_mid"]
    eq_high = np.ones(T) * params["eq_high"]

    # Loudness envelope
    loud = np.concatenate([
        np.linspace(1, 0.7, T//2),
        np.linspace(0.7, 1, T - T//2)
    ])

    loud *= (1 - 0.25 * np.exp(-2 * np.mod(beat_pos, 1)))

    return np.stack([gainA, gainB, hpf, lpf, eq_low, eq_mid, eq_high, loud], axis=-1)


################################################################################
# 2. DSP Engine
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

    def _apply_eq(self, audio, elo, emid, ehi):

        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)

        fft[freqs < 200] *= elo
        fft[(freqs >= 200) & (freqs < 2000)] *= emid
        fft[freqs >= 2000] *= ehi

        return np.fft.irfft(fft, n=len(audio))

    def render_transition(self, A, B, dsp_params):

        T = dsp_params.shape[0]
        L = min(len(A), len(B))

        x = np.linspace(0,1,T)
        xf = np.linspace(0,1,L)
        interp = lambda p: np.interp(xf, x, p)

        gainA = interp(dsp_params[:,0])
        gainB = interp(dsp_params[:,1])

        hpf_cut = interp(dsp_params[:,2])
        lpf_cut = interp(dsp_params[:,3])

        eq_low  = interp(dsp_params[:,4])
        eq_mid  = interp(dsp_params[:,5])
        eq_high = interp(dsp_params[:,6])

        loud = interp(dsp_params[:,7])

        # Per-sample filtering
        A_proc = np.zeros(L)
        B_proc = np.zeros(L)

        for i in range(L):
            A_proc[i] = self._apply_eq(
                self._filter_freq(np.array([A[i]]), hpf_cut[i], "highpass"),
                eq_low[i], eq_mid[i], eq_high[i]
            )[0]

            B_proc[i] = self._apply_eq(
                self._filter_freq(np.array([B[i]]), lpf_cut[i], "lowpass"),
                eq_low[i], eq_mid[i], eq_high[i]
            )[0]

        mix = gainA * A_proc + gainB * B_proc
        mix *= loud

        return mix / (np.max(np.abs(mix)) + 1e-9) * 0.95


################################################################################
# 3. Create transition using MODEL OUTPUTS
################################################################################
def make_transition(
    audioA_path,
    audioB_path,
    params,                 # <---- the model output dictionary
    out_path="transition.wav",
    plot_path="dsp_plot.png",
    sr=16000,
    max_transition_seconds=20
):

    # Load audio
    A, sra = sf.read(audioA_path)
    B, srb = sf.read(audioB_path)

    if A.ndim > 1: A = A.mean(axis=1)
    if B.ndim > 1: B = B.mean(axis=1)

    A = librosa.resample(A, orig_sr=sra, target_sr=sr)
    B = librosa.resample(B, orig_sr=srb, target_sr=sr)

    duration_ratio = params["duration_ratio"]
    print(f"[Transition] duration_ratio = {duration_ratio:.3f}")

    N = int(duration_ratio * max_transition_seconds * sr)

    A_cut = A[-N:]
    B_cut = B[:N]

    beat_A, bpmA, _ = compute_beat_position(A_cut, sr, 200)
    beat_B, bpmB, _ = compute_beat_position(B_cut, sr, 200)
    beat_mix = bpm_morph_scheduler(200, beat_A, beat_B)

    dsp_params = model_dsp_schedule(200, beat_mix, params)

    engine = DSPEngine(sr)
    transition = engine.render_transition(A_cut, B_cut, dsp_params)

    sf.write(out_path, transition, sr)
    print(f"[DONE] Transition saved → {out_path}")

    return transition



################################################################################
# 4. Full Song (A + transition + rest of B) — using model outputs
################################################################################
def make_full_song(
    audioA_path,
    audioB_path,
    params,
    out_path="full_mix.wav",
    sr=16000,
    max_transition_seconds=20
):

    A, sra = sf.read(audioA_path)
    B, srb = sf.read(audioB_path)

    if A.ndim > 1: A = A.mean(axis=1)
    if B.ndim > 1: B = B.mean(axis=1)

    A = librosa.resample(A, orig_sr=sra, target_sr=sr)
    B = librosa.resample(B, orig_sr=srb, target_sr=sr)

    N = int(params["duration_ratio"] * max_transition_seconds * sr)

    A_cut = A[-N:]
    B_cut = B[:N]
    B_rest = B[N:]

    beat_A, _, _ = compute_beat_position(A_cut, sr, 200)
    beat_B, _, _ = compute_beat_position(B_cut, sr, 200)
    beat_mix = bpm_morph_scheduler(200, beat_A, beat_B)

    dsp_params = model_dsp_schedule(200, beat_mix, params)

    engine = DSPEngine(sr)
    transition = engine.render_transition(A_cut, B_cut, dsp_params)

    full_mix = np.concatenate([A[:-N], transition, B_rest])
    full_mix = full_mix / (np.max(np.abs(full_mix)) + 1e-9) * 0.98

    sf.write(out_path, full_mix, sr)
    print(f"[DONE] Full song saved → {out_path}")

    return full_mix

if __name__ == "__main__":
    model_params = {
    "hpf1": 80,
    "hpf2": 350,
    "lpf1": 14000,
    "lpf2": 4500,
    "eq_low": 1.1,
    "eq_mid": 1.25,
    "eq_high": 0.85,
    "duration_ratio": 0.75
    }

    make_transition("songA.mp3", "songB.mp3", model_params)
    make_full_song("songA.mp3", "songB.mp3", model_params)
