# dsp/dsp_engine.py
import numpy as np
import soundfile as sf


class DSPEngine:
    """
    Very simple DSP engine to render transitions from DSP parameter curves.
    非常简化版 DSP 引擎，用来从 DSP 参数渲染过渡音频。
    """

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def _filter_freq(self, audio: np.ndarray, cutoff: float,
                     mode: str = "lowpass") -> np.ndarray:
        """
        Very naive frequency-domain filter.
        """
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sr)

        if mode == "lowpass":
            fft[freqs > cutoff] = 0.0
        elif mode == "highpass":
            fft[freqs < cutoff] = 0.0

        return np.fft.irfft(fft, n=len(audio))

    def _apply_eq(self, audio: np.ndarray,
                  eq_low: float, eq_mid: float, eq_high: float) -> np.ndarray:
        """
        Very simple 3-band EQ via frequency-domain gains.
        """
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sr)

        low_band = freqs < 200
        mid_band = (freqs >= 200) & (freqs < 2000)
        high_band = freqs >= 2000

        fft[low_band] *= eq_low
        fft[mid_band] *= eq_mid
        fft[high_band] *= eq_high

        return np.fft.irfft(fft, n=len(audio))

    def render_transition(self,
                          audio_A: np.ndarray,
                          audio_B: np.ndarray,
                          dsp_params: np.ndarray) -> np.ndarray:
        """
        Render transition segment given two audio buffers and DSP curves.
        audio_A, audio_B are assumed to be mono, same sr.
        dsp_params: (T, 8) from rule-based or model prediction.
        """
        T = dsp_params.shape[0]
        length = min(len(audio_A), len(audio_B))

        # Time indices in audio for each DSP frame
        idx = np.linspace(0, length - 1, T).astype(int)

        gainA = dsp_params[:, 0]
        gainB = dsp_params[:, 1]
        hpf = dsp_params[:, 2]
        lpf = dsp_params[:, 3]
        eq_low = dsp_params[:, 4]
        eq_mid = dsp_params[:, 5]
        eq_high = dsp_params[:, 6]
        loud = dsp_params[:, 7]

        # Interpolate frame-based params to sample-based
        x = np.linspace(0, 1, T)
        x_full = np.linspace(0, 1, length)

        def interp_param(p):
            return np.interp(x_full, x, p)

        gainA_full = interp_param(gainA)
        gainB_full = interp_param(gainB)
        hpf_full = interp_param(hpf)
        lpf_full = interp_param(lpf)
        eq_low_full = interp_param(eq_low)
        eq_mid_full = interp_param(eq_mid)
        eq_high_full = interp_param(eq_high)
        loud_full = interp_param(loud)

        # Apply filters with global average cutoff to simplify
        A_filt = self._filter_freq(audio_A[:length],
                                   cutoff=hpf.mean(),
                                   mode="highpass")
        B_filt = self._filter_freq(audio_B[:length],
                                   cutoff=lpf.mean(),
                                   mode="lowpass")

        # Apply EQ with average gains
        A_eq = self._apply_eq(A_filt,
                              eq_low.mean(), eq_mid.mean(), eq_high.mean())
        B_eq = self._apply_eq(B_filt,
                              eq_low.mean(), eq_mid.mean(), eq_high.mean())

        mix = gainA_full * A_eq + gainB_full * B_eq
        mix = mix * loud_full

        return mix

    def save_wav(self, path: str, audio: np.ndarray):
        sf.write(path, audio, self.sr)
