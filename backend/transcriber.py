"""
Whisper 模型封装与转写逻辑
"""
import logging
import os
import time

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger("whisperweb")


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
            logger.info("[MODEL] 模型已加载，跳过重复加载")
            return

        logger.info("[MODEL] ========== 开始加载模型 ==========")
        logger.info(f"[MODEL]   model_path   = {self.model_path}")
        logger.info(f"[MODEL]   device       = {self.device}")
        logger.info(f"[MODEL]   compute_type = {self.compute_type}")
        logger.info(f"[MODEL]   download_root= {self.download_root}")
        logger.info(f"[MODEL]   HF_ENDPOINT  = {os.environ.get('HF_ENDPOINT', '(未设置)')}")
        logger.info(f"[MODEL]   language     = {self.language}")

        if self.hf_endpoint:
            os.environ["HF_ENDPOINT"] = self.hf_endpoint
            logger.info(f"[MODEL]   使用 HuggingFace 镜像: {self.hf_endpoint}")

        t0 = time.time()
        logger.info("[MODEL]   正在创建 WhisperModel 实例...")
        self._model = WhisperModel(
            self.model_path,
            device=self.device,
            compute_type=self.compute_type,
            download_root=self.download_root
        )
        logger.info(f"[MODEL] ✅ 模型加载完成 ({time.time() - t0:.1f}s)")

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            self.load_model()
        return self._model

    def transcribe_chunk(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """转写一段音频数据，返回文本"""
        t0 = time.time()

        # 确保音频是 float32 格式
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max

        # 单声道检查
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio_dur = len(audio) / sample_rate
        logger.info(f"[MODEL] 开始转写: {audio_dur:.1f}s 音频, {len(audio)} samples, sr={sample_rate}")

        lang = None if self.language == "auto" else self.language
        logger.debug(f"[MODEL]   参数: language={lang or 'auto'}, beam_size=5, vad_filter=True")

        segments, info = self.model.transcribe(
            audio, language=lang,
            beam_size=5, vad_filter=True
        )
        logger.debug(f"[MODEL]   检测语言: {info.language} (概率: {info.language_probability:.2f})")

        texts = [seg.text.strip() for seg in segments]
        result = " ".join(texts)

        logger.info(f"[MODEL] 转写完成 ({time.time() - t0:.1f}s): "
                    f"语言={info.language}, "
                    f"段数={len(texts)}, "
                    f"文本长度={len(result)}")

        return result

    def update_config(self, model_path: str = None, device: str = None,
                      compute_type: str = None, language: str = None,
                      download_root: str = None, hf_endpoint: str = None):
        """更新模型配置，如果需要会重新加载模型"""
        logger.info("[MODEL] 检查配置更新...")
        need_reload = False
        if model_path and model_path != self.model_path:
            logger.info(f"[MODEL]   model_path 变更: {self.model_path} -> {model_path}")
            self.model_path = model_path
            need_reload = True
        if device and device != self.device:
            logger.info(f"[MODEL]   device 变更: {self.device} -> {device}")
            self.device = device
            need_reload = True
        if compute_type and compute_type != self.compute_type:
            logger.info(f"[MODEL]   compute_type 变更: {self.compute_type} -> {compute_type}")
            self.compute_type = compute_type
            need_reload = True
        if download_root and download_root != self.download_root:
            logger.info(f"[MODEL]   download_root 变更: {self.download_root} -> {download_root}")
            self.download_root = download_root
            need_reload = True
        if hf_endpoint is not None and hf_endpoint != self.hf_endpoint:
            logger.info(f"[MODEL]   hf_endpoint 变更: {self.hf_endpoint} -> {hf_endpoint}")
            self.hf_endpoint = hf_endpoint
            need_reload = True
        if language is not None:
            logger.info(f"[MODEL]   language 变更: {self.language} -> {language}")
            self.language = language

        if need_reload:
            logger.info("[MODEL] 配置已变更，重新加载模型...")
            self._model = None
            self.load_model()
        else:
            logger.info("[MODEL] 配置未变更，无需重载")
