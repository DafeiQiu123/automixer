import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt


################################################################################
# 0. Beat Position (BPM + absolute beat position)
################################################################################

def bpm_morph_scheduler(T, beat_pos_A, beat_pos_B):
    """
    Smoothly interpolate beat positions between song A and B.
    t=0 → follow A
    t=1 → follow B
    """
    t = np.linspace(0, 1, T)
    return (1 - t) * beat_pos_A + t * beat_pos_B

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

    # Half/double time correction
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



################################################################################
# 1. DSP Scheduler (Uses model output parameters)
################################################################################
def model_dsp_schedule(T, beat_pos, params):
    """
    Input:
        T          - number of DSP steps
        beat_pos   - absolute beat positions
        params     - dict from model output:
            hpf1, hpf2
            lpf1, lpf2
            eq_low, eq_mid, eq_high
            duration_ratio (not used here)
    """

    t = np.linspace(0, 1, T)

    # Crossfade gains (same as default system)
    gainA = (1 - t) + 0.05 * np.sin(2 * np.pi * beat_pos)
    gainB = t         + 0.05 * np.sin(2 * np.pi * beat_pos)

    # Filters use simple two-stage morphing
    stage = (t >= 0.5).astype(float)

    hpf = params["hpf1"] * (1 - stage) + params["hpf2"] * stage
    lpf = params["lpf1"] * (1 - stage) + params["lpf2"] * stage

    # EQ is constant during transition
    eq_low  = np.ones(T) * params["eq_low"]
    eq_mid  = np.ones(T) * params["eq_mid"]
    eq_high = np.ones(T) * params["eq_high"]

    # Loudness envelope
    loud = np.concatenate([
        np.linspace(1.0, 0.7, T//2),
        np.linspace(0.7, 1.0, T - T//2)
    ])

    loud *= (1 - 0.25 * np.exp(-2 * np.mod(beat_pos, 1)))

    # Final DSP matrix
    return np.stack([
        gainA, gainB,
        hpf, lpf,
        eq_low, eq_mid, eq_high,
        loud
    ], axis=-1)



################################################################################
# 2. DSP Engine (Correct block-based DSP implementation)
################################################################################
class DSPEngine:
    def __init__(self, sr=16000):
        self.sr = sr

    # ================================
    #   Frequency-domain HPF/LPF
    # ================================
    def _filter_freq(self, audio, cutoff, mode="lowpass"):
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1 / self.sr)

        if mode == "lowpass":
            fft[freqs > cutoff] = 0.0
        else:
            fft[freqs < cutoff] = 0.0

        return np.fft.irfft(fft, n=len(audio))

    # ================================
    #   Simple 3-band EQ
    # ================================
    def _apply_eq(self, audio, elo, emid, ehi):

        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)

        fft[freqs < 200] *= elo
        fft[(freqs >= 200) & (freqs < 2000)] *= emid
        fft[freqs >= 2000] *= ehi

        return np.fft.irfft(fft, n=len(audio))


    # ================================
    #   Correct transition renderer
    # ================================
    def render_transition(self, A, B, dsp_params):

        T = dsp_params.shape[0]
        L = min(len(A), len(B))

        x = np.linspace(0, 1, T)
        xf = np.linspace(0, 1, L)
        interp = lambda p: np.interp(xf, x, p)

        # Gains (time-varying)
        gainA = interp(dsp_params[:, 0])
        gainB = interp(dsp_params[:, 1])

        # Cutoffs → must use averaged values (block-based DSP)
        hpf_cut = float(np.mean(dsp_params[:, 2]))
        lpf_cut = float(np.mean(dsp_params[:, 3]))

        # EQ → also block-based
        eq_low  = float(np.mean(dsp_params[:, 4]))
        eq_mid  = float(np.mean(dsp_params[:, 5]))
        eq_high = float(np.mean(dsp_params[:, 6]))

        loud = interp(dsp_params[:, 7])

        # Filtering whole segments
        A_f = self._filter_freq(A[:L], hpf_cut, mode="highpass")
        B_f = self._filter_freq(B[:L], lpf_cut, mode="lowpass")

        A_p = self._apply_eq(A_f, eq_low, eq_mid, eq_high)
        B_p = self._apply_eq(B_f, eq_low, eq_mid, eq_high)

        # Mix
        mix = gainA * A_p + gainB * B_p
        mix *= loud

        return mix / (np.max(np.abs(mix)) + 1e-9) * 0.95



################################################################################
# 3. Make Transition (using model-predicted parameters)
################################################################################
def make_transition(
    audioA_path,
    audioB_path,
    params,                 # model output dict
    out_path="transition.wav",
    plot_path="dsp_plot.png",
    sr=16000,
    max_transition_seconds=15
):

    A, sra = sf.read(audioA_path)
    B, srb = sf.read(audioB_path)

    if A.ndim > 1: A = A.mean(axis=1)
    if B.ndim > 1: B = B.mean(axis=1)

    A = librosa.resample(A, orig_sr=sra, target_sr=sr)
    B = librosa.resample(B, orig_sr=srb, target_sr=sr)

    # Transition segment length
    duration_ratio = params["duration_ratio"]
    N = int(duration_ratio * max_transition_seconds * sr)

    A_cut = A[-N:]
    B_cut = B[:N]

    # Beat positions for morphing
    beat_A, _, _ = compute_beat_position(A_cut, sr, 200)
    beat_B, _, _ = compute_beat_position(B_cut, sr, 200)

    beat_mix = bpm_morph_scheduler(200, beat_A, beat_B)

    dsp_params = model_dsp_schedule(200, beat_mix, params)

    engine = DSPEngine(sr)
    transition = engine.render_transition(A_cut, B_cut, dsp_params)

    sf.write(out_path, transition, sr)
    print(f"[DONE] Transition saved → {out_path}")

    return transition



################################################################################
# 4. Full Song (A + transition + B_rest)
################################################################################
def make_full_song(
    audioA_path,
    audioB_path,
    params,
    out_path="full_mix.wav",
    sr=16000,
    max_transition_seconds=15
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



################################################################################
# Example
################################################################################
if __name__ == "__main__":

    model_params = {
        "hpf1": 150.04962158203125,
  "hpf2": 270.0126647949219,
  "lpf1": 7992.556640625,
  "lpf2": 12000.0,
  "eq_low": 1.2662874460220337,
  "eq_mid": 1.0064622163772583,
  "eq_high": 0.648768424987793,
  "duration_ratio": 0.34610041975975037,
    }

    make_transition("songA_1.wav", "songB_1.wav", model_params)
    make_full_song("songA_1.wav", "songB_1.wav", model_params)
