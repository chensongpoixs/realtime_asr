"""
Whisper 模型封装与转写逻辑
"""
import os
import numpy as np
from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_path: str = "medium", device: str = "cpu",
                 compute_type: str = "int8", language: str = "auto",
                 download_root: str = "./models", hf_endpoint: str = ""):
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.download_root = download_root
        self.hf_endpoint = hf_endpoint
        self._model: WhisperModel | None = None

    def load_model(self):
        """加载 Whisper 模型"""
        if self._model is not None:
            return
        if self.hf_endpoint:
            os.environ["HF_ENDPOINT"] = self.hf_endpoint
            import logging
            logging.getLogger("whisperweb").info(
                f"[MODEL] 使用 HuggingFace 镜像: {self.hf_endpoint}"
            )
        self._model = WhisperModel(
            self.model_path,
            device=self.device,
            compute_type=self.compute_type,
            download_root=self.download_root
        )

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            self.load_model()
        return self._model

    def transcribe_chunk(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """转写一段音频数据，返回文本"""
        # 确保音频是 float32 格式
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max

        # 单声道检查
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        lang = None if self.language == "auto" else self.language
        segments, _ = self.model.transcribe(
            audio, language=lang,
            beam_size=5, vad_filter=True
        )

        texts = [seg.text.strip() for seg in segments]
        return " ".join(texts)

    def update_config(self, model_path: str = None, device: str = None,
                      compute_type: str = None, language: str = None,
                      download_root: str = None, hf_endpoint: str = None):
        """更新模型配置，如果需要会重新加载模型"""
        need_reload = False
        if model_path and model_path != self.model_path:
            self.model_path = model_path
            need_reload = True
        if device and device != self.device:
            self.device = device
            need_reload = True
        if compute_type and compute_type != self.compute_type:
            self.compute_type = compute_type
            need_reload = True
        if download_root and download_root != self.download_root:
            self.download_root = download_root
            need_reload = True
        if hf_endpoint is not None and hf_endpoint != self.hf_endpoint:
            self.hf_endpoint = hf_endpoint
            need_reload = True
        if language is not None:
            self.language = language

        if need_reload:
            self._model = None
            self.load_model()
