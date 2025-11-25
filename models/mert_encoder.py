# models/mert_encoder.py
import torch
import torchaudio
import numpy as np
from transformers import AutoModel, AutoProcessor
from scipy.interpolate import interp1d


class MERTEncoder:
    """
    Wrap MERT to get frame-level embeddings for music segments.
    使用 MERT 提取帧级音乐表征。
    """

    def __init__(self, model_name: str = "m-a-p/MERT-v1-95M"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True
        ).to(self.device)

        self.target_sr = 16000
        self.embedding_dim = self.model.config.hidden_size  # e.g., 768

    def load_audio(self, path: str, start_sec: float | None = None,
                   end_sec: float | None = None) -> torch.Tensor:
        """
        Load audio, convert to mono, resample to target_sr, and optional trim.
        加载音频，转 mono，重采样，截取片段。
        """
        audio, sr = torchaudio.load(path)  # (C, T)

        # mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        if sr != self.target_sr:
            audio = torchaudio.functional.resample(audio, sr, self.target_sr)

        audio = audio.squeeze(0)  # (T,)

        if start_sec is not None and end_sec is not None:
            start = int(start_sec * self.target_sr)
            end = int(end_sec * self.target_sr)
            audio = audio[start:end]

        return audio

    def extract_embeddings(self, audio_tensor: torch.Tensor) -> np.ndarray:
        """
        Run MERT and get frame-level embeddings.
        返回形状 (T_mert, d) 的帧级 embedding。
        """
        inputs = self.processor(
            audio_tensor,
            sampling_rate=self.target_sr,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use second-to-last hidden layer as canonical MERT embedding
        hidden = outputs.hidden_states[-2].squeeze(0)  # (T, d)
        return hidden.cpu().numpy()

    def resample_frames(self, frames: np.ndarray, target_frames: int) -> np.ndarray:
        """
        Resample frame sequence to a fixed length along time.
        将时间维度重采样到固定帧数。
        """
        T, D = frames.shape
        if T == target_frames:
            return frames

        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, target_frames)
        f = interp1d(x_old, frames, axis=0, kind="linear")
        return f(x_new)

    def encode_segment(self, path: str, start_sec: float | None,
                       end_sec: float | None, target_frames: int) -> np.ndarray:
        """
        High-level API: audio file -> segment -> MERT -> fixed-length frames.
        """
        audio = self.load_audio(path, start_sec, end_sec)
        emb = self.extract_embeddings(audio)
        emb_resampled = self.resample_frames(emb, target_frames)
        return emb_resampled  # (T, d)
