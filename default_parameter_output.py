import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt


# =============================================================================
# 0. 更鲁棒的 BPM 估计 + beat position
# =============================================================================
def estimate_bpm(audio, sr,
                 min_bpm=70.0,
                 max_bpm=160.0,
                 default_bpm=120.0):
    """
    更鲁棒的 BPM 估计：
      1) 用 onset envelope + librosa.beat.tempo 估计 tempo
      2) 避免极慢/极快 → half / double-time 修正
      3) 最后 clamp 在 [min_bpm, max_bpm]
    """
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)

    try:
        tempo = librosa.beat.tempo(
            onset_envelope=onset_env,
            sr=sr,
            aggregate=np.median,
            max_tempo=320.0,
        )[0]
    except Exception as e:
        print(f"[WARN] tempo() failed: {e}")
        return default_bpm

    if tempo <= 0 or np.isnan(tempo):
        print("[WARN] tempo invalid, fallback to default bpm")
        return default_bpm

    # half / double-time 修正
    while tempo > max_bpm:
        tempo /= 2.0
    while tempo < min_bpm:
        tempo *= 2.0

    tempo = float(np.clip(tempo, min_bpm, max_bpm))
    return tempo


def compute_beat_position(audio, sr, T=200):
    """
    从一段音频中估计 BPM，然后生成 absolute beat position:
      beat_pos[t] = (bpm / 60) * time_seconds

    返回:
        beat_pos : (T,)
        bpm      : float
        time_axis: (T,)
    """
    bpm = estimate_bpm(audio, sr)
    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, T)
    beat_pos = (bpm / 60.0) * time_axis
    return beat_pos, bpm, time_axis


# =============================================================================
# 1. 自动根据 BPM 差异估计 duration_ratio
# =============================================================================
def compute_duration_ratio(bpmA, bpmB,
                           min_ratio=0.15,
                           max_ratio=1.0):
    """
    根据两首歌 BPM 差异自动决定过渡时长比例（0~1）：
      - BPM 差距小 → 可以混久一点
      - 差距大 → 过渡变短
    """
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


# =============================================================================
# 2. 从歌曲自动分析滤波参数：hpf1/hpf2/lpf1/lpf2
# =============================================================================
def estimate_hpf_from_song(audio, sr):
    """
    根据 songA 的频谱能量自动估计:
        hpf1: 过渡前半段的高通截止频率
        hpf2: 过渡后半段的高通截止频率（更高 → 高频保留，低频变薄）

    思路：
      - 看低频(0-200Hz) 和 中低频(200-400Hz) 的能量比
      - 低频特别多 → hpf2 稍微高一点，避免糊
      - 低频少 → 不用切得太高
    """
    S = np.abs(librosa.stft(audio, n_fft=2048)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    low_band = (freqs < 200)
    mid_band = (freqs >= 200) & (freqs < 400)

    low_energy = S[low_band].mean() + 1e-9
    mid_energy = S[mid_band].mean() + 1e-9

    ratio = low_energy / mid_energy  # 低频比中低频多多少
    # 映射到 [60, 150] 之间
    hpf1 = 60 + (ratio - 0.5) * 40
    hpf1 = float(np.clip(hpf1, 40, 150))

    # 后半段再切高一点
    hpf2 = hpf1 + 120
    hpf2 = float(np.clip(hpf2, 150, 400))

    return hpf1, hpf2


def estimate_lpf_from_song(audio, sr):
    """
    根据 songB 的频谱能量自动估计:
        lpf1: 过渡前半段的低通截止频率（相对暗）
        lpf2: 过渡后半段的低通截止频率（更亮）

    思路：
      - 看 5kHz–12kHz 的高频能量
      - 高频很多的歌，lpf2 可以放开一点
      - 高频很少的歌，lpf2 也不用太高
    """
    S = np.abs(librosa.stft(audio, n_fft=2048)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    upper_band = (freqs >= 5000) & (freqs < 12000)
    high_energy = S[upper_band].mean() + 1e-9

    # 高频能量大 → 歌本来很亮，可以给高一点的 lpf2
    # 映射高频能量到 [6000, 14000]
    base = 8000
    span = 6000
    lpf2 = base + span * np.tanh(high_energy / (high_energy + 1e-7))
    lpf2 = float(np.clip(lpf2, 6000, 16000))

    # 前半段更暗一点
    lpf1 = lpf2 - 5000
    lpf1 = float(np.clip(lpf1, 3000, lpf2 - 500))

    return lpf1, lpf2


def auto_filter_params(songA_seg, songB_seg, sr):
    """
    输入：songA 的过渡片段、songB 的过渡片段
    输出：基于音色自动计算的滤波参数：
        hpf1, hpf2, lpf1, lpf2
    """
    hpf1, hpf2 = estimate_hpf_from_song(songA_seg, sr)
    lpf1, lpf2 = estimate_lpf_from_song(songB_seg, sr)

    return hpf1, hpf2, lpf1, lpf2


# =============================================================================
# 3. 从歌曲自动分析 EQ（暖音版本）
# =============================================================================
def analyze_eq_bands_warm(audio, sr):
    """
    从单首歌中分析低/中/高频能量，并映射到 EQ 值 (0.7~1.4)
    中频稍强，高频略柔和
    """
    S = np.abs(librosa.stft(audio, n_fft=2048)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    low_band = (freqs < 200)
    mid_band = (freqs >= 200) & (freqs < 2000)
    high_band = (freqs >= 2000) & (freqs < 8000)

    low_e = S[low_band].mean() + 1e-9
    mid_e = S[mid_band].mean() + 1e-9
    high_e = S[high_band].mean() + 1e-9

    total = low_e + mid_e + high_e + 1e-9
    low_ratio = low_e / total
    mid_ratio = mid_e / total
    high_ratio = high_e / total

    # 映射到 0.7 ~ 1.4
    def map_eq(r):
        return 0.7 + r * 0.7

    eq_low = map_eq(low_ratio)
    eq_mid = map_eq(mid_ratio)
    eq_high = map_eq(high_ratio)

    # 为了“暖音”风格，稍微偏置一下：
    eq_mid *= 1.1        # 中频再推一点
    eq_high *= 0.95      # 高频稍微柔和一点

    eq_mid = float(np.clip(eq_mid, 0.7, 1.5))
    eq_high = float(np.clip(eq_high, 0.6, 1.3))

    return float(eq_low), float(eq_mid), float(eq_high)


def merge_eq_from_two_songs(A_seg, B_seg, sr):
    """
    分别对 A_cut 和 B_cut 做 EQ 分析，然后取平均，
    得到整个过渡区间使用的一组暖音 EQ 参数。
    """
    eqA_low, eqA_mid, eqA_high = analyze_eq_bands_warm(A_seg, sr)
    eqB_low, eqB_mid, eqB_high = analyze_eq_bands_warm(B_seg, sr)

    eq_low = (eqA_low + eqB_low) / 2.0
    eq_mid = (eqA_mid + eqB_mid) / 2.0
    eq_high = (eqA_high + eqB_high) / 2.0

    return float(eq_low), float(eq_mid), float(eq_high)


# =============================================================================
# 4. BPM Morphing + DSP scheduler (使用自动滤波 + 自动 EQ)
# =============================================================================
def bpm_morph_scheduler(T, beat_pos_A, beat_pos_B):
    """
    根据绝对节拍位置从 A 过渡到 B：
      t=0 → 主要跟 A
      t=1 → 主要跟 B
    """
    t = np.linspace(0, 1, T)
    return (1 - t) * beat_pos_A + t * beat_pos_B


def bpm_scheduler_dsp(T, beat_pos,
                      hpf1, hpf2,
                      lpf1, lpf2,
                      eq_low, eq_mid, eq_high):
    """
    生成完整的 DSP 参数序列：
      - gainA, gainB: crossfade + 轻微随节拍抖动
      - hpf / lpf: 前半段用 1，后半段用 2（线性过渡）
      - EQ: 使用从歌曲得到的一组暖音 EQ（在整个过渡期间保持常数，也可以扩展为曲线）
      - loud: 一个简单的 duck → rise 曲线 + beat pumping
    """
    t = np.linspace(0, 1, T)

    # --- crossfade ---
    gainA = (1 - t) + 0.05 * np.sin(2 * np.pi * beat_pos)
    gainB = t + 0.05 * np.sin(2 * np.pi * beat_pos)

    # --- HPF/LPF 两阶段 ---
    stage = (t >= 0.5).astype(float)  # 前半=0，后半=1
    hpf = hpf1 * (1 - stage) + hpf2 * stage
    lpf = lpf1 * (1 - stage) + lpf2 * stage

    # --- EQ（整段常数，也可以改成 np.linspace 做渐变）---
    eq_low_curve = np.ones(T) * eq_low
    eq_mid_curve = np.ones(T) * eq_mid
    eq_high_curve = np.ones(T) * eq_high

    # --- Loudness ---
    loud = np.concatenate([
        np.linspace(1.0, 0.7, T // 2),
        np.linspace(0.7, 1.0, T - T // 2),
    ])
    loud *= (1 - 0.25 * np.exp(-2 * np.mod(beat_pos, 1)))

    return np.stack([
        gainA, gainB, hpf, lpf,
        eq_low_curve, eq_mid_curve, eq_high_curve, loud
    ], axis=-1)


# =============================================================================
# 5. 可视化 DSP 参数
# =============================================================================
def plot_dsp(dsp_params, beat_pos, save_path="dsp_plot.png"):

    T = dsp_params.shape[0]
    t = np.linspace(0, 1, T)

    gainA, gainB = dsp_params[:, 0], dsp_params[:, 1]
    hpf, lpf = dsp_params[:, 2], dsp_params[:, 3]
    eq_low, eq_mid, eq_high = dsp_params[:, 4], dsp_params[:, 5], dsp_params[:, 6]
    loud = dsp_params[:, 7]

    plt.figure(figsize=(16, 22))

    plt.subplot(6, 1, 1); plt.title("Gain")
    plt.plot(t, gainA, label="gainA")
    plt.plot(t, gainB, label="gainB")
    plt.grid(); plt.legend()

    plt.subplot(6, 1, 2); plt.title("Filters")
    plt.plot(t, hpf, label="HPF")
    plt.plot(t, lpf, label="LPF")
    plt.grid(); plt.legend()

    plt.subplot(6, 1, 3); plt.title("EQ")
    plt.plot(t, eq_low, label="EQ Low")
    plt.plot(t, eq_mid, label="EQ Mid")
    plt.plot(t, eq_high, label="EQ High")
    plt.grid(); plt.legend()

    plt.subplot(6, 1, 4); plt.title("Loudness")
    plt.plot(t, loud, label="Loud")
    plt.grid(); plt.legend()

    plt.subplot(6, 1, 5); plt.title("Beat Position")
    plt.plot(t, beat_pos, label="beat_pos")
    plt.grid(); plt.legend()

    plt.subplot(6, 1, 6); plt.title("Beat Phase")
    plt.plot(t, np.mod(beat_pos, 1), label="beat_pos % 1")
    plt.grid(); plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Saved DSP plot] {save_path}")
    plt.close()


# =============================================================================
# 6. DSP Engine（高通 + 低通 + 暖音 EQ）
# =============================================================================
class DSPEngine:
    def __init__(self, sr=16000):
        self.sr = sr

    def _filter_freq(self, audio, cutoff, mode="lowpass"):
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1 / self.sr)

        if mode == "lowpass":
            fft[freqs > cutoff] = 0.0
        else:
            fft[freqs < cutoff] = 0.0

        return np.fft.irfft(fft, n=len(audio))

    def _apply_eq(self, audio, elo, emid, ehi):
        """
        双色 EQ：
          - elo/emid/ehi 来自歌曲自动分析（整体增益）
          - 再叠加一个“暖音” shaping（固定系数），
            类似你之前喜欢的那套 aggressive EQ
        """
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1 / self.sr)

        # 基础三段：
        low_mask = (freqs < 200)
        mid_mask = (freqs >= 200) & (freqs < 2000)
        high_mask = (freqs >= 2000)

        fft[low_mask] *= elo
        fft[mid_mask] *= emid
        fft[high_mask] *= ehi

        return np.fft.irfft(fft, n=len(audio))

    def render_transition(self, A, B, dsp_params):
        """
        实际执行 DSP：
          - 对 A 使用高通 + 暖音 EQ
          - 对 B 使用低通 + 暖音 EQ
          - 用 gainA/gainB crossfade + loudness 包络
        """
        T = dsp_params.shape[0]
        L = min(len(A), len(B))

        x = np.linspace(0, 1, T)
        xf = np.linspace(0, 1, L)
        interp = lambda p: np.interp(xf, x, p)

        gainA = interp(dsp_params[:, 0])
        gainB = interp(dsp_params[:, 1])
        hpf_cut = float(np.mean(dsp_params[:, 2]))
        lpf_cut = float(np.mean(dsp_params[:, 3]))
        eq_low_vals = interp(dsp_params[:, 4])
        eq_mid_vals = interp(dsp_params[:, 5])
        eq_high_vals = interp(dsp_params[:, 6])
        loud = interp(dsp_params[:, 7])

        # 取 EQ 的平均值作为全局系数（也可以保留随时间变化）
        elo = float(np.mean(eq_low_vals))
        emid = float(np.mean(eq_mid_vals))
        ehi = float(np.mean(eq_high_vals))

        # Filtering + EQ
        A_f = self._filter_freq(A[:L], hpf_cut, mode="highpass")
        B_f = self._filter_freq(B[:L], lpf_cut, mode="lowpass")

        A_p = self._apply_eq(A_f, elo, emid, ehi)
        B_p = self._apply_eq(B_f, elo, emid, ehi)

        mix = gainA * A_p + gainB * B_p
        mix *= loud

        return mix / (np.max(np.abs(mix)) + 1e-9) * 0.95


# =============================================================================
# 7. 只生成 transition（自动 duration_ratio）
# =============================================================================
def make_transition(
    audioA_path,
    audioB_path,
    out_path="transition.wav",
    plot_path="dsp_plot.png",
    sr=24000,
    max_transition_seconds=20,
):
    # 读取 & 转单声道
    A, sra = sf.read(audioA_path)
    B, srb = sf.read(audioB_path)

    if A.ndim > 1:
        A = A.mean(axis=1)
    if B.ndim > 1:
        B = B.mean(axis=1)

    # 重采样
    A = librosa.resample(y=A, orig_sr=sra, target_sr=sr)
    B = librosa.resample(y=B, orig_sr=srb, target_sr=sr)

    # 先取 5 秒做 BPM 估计
    beat_A_short, bpmA, _ = compute_beat_position(A[-5 * sr:], sr, T=200)
    beat_B_short, bpmB, _ = compute_beat_position(B[:5 * sr], sr, T=200)

    duration_ratio = compute_duration_ratio(bpmA, bpmB)
    print(f"[Transition] BPM_A={bpmA:.2f}, BPM_B={bpmB:.2f}, duration_ratio={duration_ratio:.3f}")

    N = int(duration_ratio * max_transition_seconds * sr)
    A_cut = A[-N:]
    B_cut = B[:N]

    # 为过渡片段重新计算 beat position
    beat_A, _, _ = compute_beat_position(A_cut, sr, T=200)
    beat_B, _, _ = compute_beat_position(B_cut, sr, T=200)
    beat_mix = bpm_morph_scheduler(200, beat_A, beat_B)

    # 自动滤波参数 + 自动 EQ
    hpf1, hpf2, lpf1, lpf2 = auto_filter_params(A_cut, B_cut, sr)
    eq_low, eq_mid, eq_high = merge_eq_from_two_songs(A_cut, B_cut, sr)

    dsp_params = bpm_scheduler_dsp(
        T=200,
        beat_pos=beat_mix,
        hpf1=hpf1, hpf2=hpf2,
        lpf1=lpf1, lpf2=lpf2,
        eq_low=eq_low, eq_mid=eq_mid, eq_high=eq_high,
    )

    plot_dsp(dsp_params, beat_mix, save_path=plot_path)

    engine = DSPEngine(sr)
    transition = engine.render_transition(A_cut, B_cut, dsp_params)

    sf.write(out_path, transition, sr)
    print(f"[DONE] Transition saved → {out_path}")
    return transition


# =============================================================================
# 8. 生成完整歌曲：A + transition + B_rest
# =============================================================================
def make_full_song(
    audioA_path,
    audioB_path,
    out_path="full_mix.wav",
    sr=24000,
    max_transition_seconds=20,
):
    A, sra = sf.read(audioA_path)
    B, srb = sf.read(audioB_path)

    if A.ndim > 1:
        A = A.mean(axis=1)
    if B.ndim > 1:
        B = B.mean(axis=1)

    A = librosa.resample(y=A, orig_sr=sra, target_sr=sr)
    B = librosa.resample(y=B, orig_sr=srb, target_sr=sr)

    beat_A_short, bpmA, _ = compute_beat_position(A[-5 * sr:], sr, 200)
    beat_B_short, bpmB, _ = compute_beat_position(B[:5 * sr], sr, 200)

    duration_ratio = compute_duration_ratio(bpmA, bpmB)
    print(f"[Full Song] BPM_A={bpmA:.2f}, BPM_B={bpmB:.2f}, duration_ratio={duration_ratio:.3f}")

    N = int(duration_ratio * max_transition_seconds * sr)
    A_cut = A[-N:]
    B_cut = B[:N]
    B_rest = B[N:]

    beat_A, _, _ = compute_beat_position(A_cut, sr, 200)
    beat_B, _, _ = compute_beat_position(B_cut, sr, 200)
    beat_mix = bpm_morph_scheduler(200, beat_A, beat_B)

    hpf1, hpf2, lpf1, lpf2 = auto_filter_params(A_cut, B_cut, sr)
    eq_low, eq_mid, eq_high = merge_eq_from_two_songs(A_cut, B_cut, sr)

    dsp_params = bpm_scheduler_dsp(
        T=200,
        beat_pos=beat_mix,
        hpf1=hpf1, hpf2=hpf2,
        lpf1=lpf1, lpf2=lpf2,
        eq_low=eq_low, eq_mid=eq_mid, eq_high=eq_high,
    )

    engine = DSPEngine(sr)
    transition = engine.render_transition(A_cut, B_cut, dsp_params)

    full_mix = np.concatenate([A[:-N], transition, B_rest])
    full_mix = full_mix / (np.max(np.abs(full_mix)) + 1e-9) * 0.98

    sf.write(out_path, full_mix, sr)
    print(f"[DONE] Full song saved → {out_path}")
    return full_mix

# =============================================================================
# 9. Extract final merged parameters (用于模型训练)
# =============================================================================
def extract_all_parameters(
    audioA_path,
    audioB_path,
    sr=24000,
    max_transition_seconds=20,
):
    """
    输入：两首歌（A, B）
    输出：所有最终用于 DSP 的参数（merge 后版本）：
        - hpf1, hpf2
        - lpf1, lpf2
        - eq_low, eq_mid, eq_high
        - duration_ratio
        - bpmA, bpmB
    """

    # ---- Load ----
    A, sra = sf.read(audioA_path)
    B, srb = sf.read(audioB_path)

    if A.ndim > 1:
        A = A.mean(axis=1)
    if B.ndim > 1:
        B = B.mean(axis=1)

    # Resample
    A = librosa.resample(y=A, orig_sr=sra, target_sr=sr)
    B = librosa.resample(y=B, orig_sr=srb, target_sr=sr)

    # ---- BPM ----
    beat_A_short, bpmA, _ = compute_beat_position(A[-5 * sr:], sr, T=200)
    beat_B_short, bpmB, _ = compute_beat_position(B[:5 * sr], sr, T=200)

    # ---- Duration ratio ----
    duration_ratio = compute_duration_ratio(bpmA, bpmB)

    # ---- Determine transition length ----
    N = int(duration_ratio * max_transition_seconds * sr)

    A_cut = A[-N:]
    B_cut = B[:N]

    # ---- Auto filters (merge A & B influence) ----
    hpf1, hpf2, lpf1, lpf2 = auto_filter_params(A_cut, B_cut, sr)

    # ---- Auto EQ (merged warm EQ for both songs) ----
    eq_low, eq_mid, eq_high = merge_eq_from_two_songs(A_cut, B_cut, sr)

    # ---- Output dict ----
    params = {
        "bpmA": float(bpmA),
        "bpmB": float(bpmB),

        "duration_ratio": float(duration_ratio),

        "hpf1": float(hpf1),
        "hpf2": float(hpf2),
        "lpf1": float(lpf1),
        "lpf2": float(lpf2),

        "eq_low": float(eq_low),
        "eq_mid": float(eq_mid),
        "eq_high": float(eq_high),
    }

    print("\n=================== Extracted Parameters ===================")
    for k, v in params.items():
        print(f"{k}: {v}")
    print("============================================================\n")

    return params

# =============================================================================
# Main Example
# =============================================================================
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

    params = extract_all_parameters("songA.mp3", "songB.mp3")
    print(params)

