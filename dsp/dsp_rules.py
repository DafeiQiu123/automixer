# dsp/dsp_rules.py
import numpy as np


def rule_based_dsp(target_frames: int = 200) -> np.ndarray:
    """
    Generate rule-based DSP curves for a transition of length T frames.
    用规则生成一条过渡用的 DSP 参数轨迹。
    返回 shape: (T, 8)
      [gainA, gainB, HPF, LPF, EQ_low, EQ_mid, EQ_high, loudness]
    """
    T = target_frames
    t = np.linspace(0, 1, T)

    # 1) Crossfade: linear A->B
    gainA = 1.0 - t
    gainB = t

    # 2) HPF (A gets more high-passed over time)
    hpf = 50.0 + 3000.0 * t  # Hz

    # 3) LPF (B gets less low-passed over time)
    lpf = 12000.0 - 6000.0 * t  # Hz

    # 4) EQ: simple "mid bump" like EDM buildup
    eq_low = 1.0 - 0.2 * np.sin(np.pi * t)
    eq_mid = 1.0 + 0.5 * np.sin(np.pi * t)
    eq_high = 1.0 + 0.2 * np.sin(np.pi * t)

    # 5) Loudness: slight swell in the middle
    loudness = 1.0 + 0.1 * np.sin(np.pi * t)

    dsp = np.stack(
        [gainA, gainB, hpf, lpf, eq_low, eq_mid, eq_high, loudness],
        axis=-1
    )  # (T, 8)
    return dsp
