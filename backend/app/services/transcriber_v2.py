"""
Triton Inference Server 异步转写引擎 (v2.0)

使用 Triton gRPC 客户端与 ONNX Runtime + TensorRT 后端通信，
实现非阻塞、支持 dynamic batching 的高并发实时转写。

架构:
    audio (Int16 PCM) → mel spectrogram (NumPy)
    → Triton Encoder (ONNX + TRT FP16, dynamic batching)
    → Triton Decoder (ONNX + TRT FP16, autoregressive loop)
    → token IDs → text

线程安全: 每个 WebSocket 连接独立调用 transcribe_chunk()，
         所有 gRPC 调用通过 asyncio 并发执行。
"""
import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("realtime_asr.v2")


# ═══════════════════════════════════════════════════════════════
# Mel Spectrogram 转换 (纯 NumPy，无 torch 运行时依赖)
# ═══════════════════════════════════════════════════════════════

def _create_mel_filters(
    n_mels: int = 80,
    n_fft: int = 400,
    sample_rate: int = 16000,
) -> np.ndarray:
    """创建 mel 滤波器组矩阵，与 OpenAI Whisper 的 mel filterbank 一致。

    Returns:
        shape (n_mels, n_fft // 2 + 1) 的 float32 数组
    """
    # Hz 转 mel 刻度
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    f_min, f_max = 0.0, float(sample_rate) / 2.0
    mel_min, mel_max = hz_to_mel(f_min), hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # FFT bin 频率
    bin_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    # 构建三角滤波器
    filters = np.zeros((n_mels, len(bin_freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        # 上升沿
        mask = (bin_freqs >= lo) & (bin_freqs <= mid)
        if mid > lo:
            filters[i, mask] = (bin_freqs[mask] - lo) / (mid - lo)
        # 下降沿
        mask = (bin_freqs >= mid) & (bin_freqs <= hi)
        if hi > mid:
            filters[i, mask] = (hi - bin_freqs[mask]) / (hi - mid)

    # 归一化 (Slaney-style)
    enorm = 2.0 / (hz_points[2:n_mels + 2] - hz_points[:n_mels])
    filters *= enorm[:, np.newaxis]

    return filters


def _load_mel_filters() -> np.ndarray:
    """加载 mel 滤波器组，优先从 .npy 文件加载，否则运行时计算。

    .npy 文件由 export_whisper_to_onnx.py 脚本导出，
    包含从 Whisper 模型权重中提取的精确 mel filterbank。
    """
    # 搜索路径
    search_paths = [
        Path(__file__).parent.parent.parent / "triton_model_repo" / "mel_filters.npy",
        Path(__file__).parent.parent.parent / "scripts" / "mel_filters.npy",
    ]

    for p in search_paths:
        if p.exists():
            filters = np.load(p)
            logger.info("[MODEL] 加载 mel filterbank: %s (shape=%s)", p, filters.shape)
            return filters.astype(np.float32)

    # 运行时计算 (备选)
    logger.info("[MODEL] mel_filters.npy 未找到，运行时计算 mel filterbank")
    return _create_mel_filters()


# 模块级缓存
_MEL_FILTERS: Optional[np.ndarray] = None


def _get_mel_filters() -> np.ndarray:
    global _MEL_FILTERS
    if _MEL_FILTERS is None:
        _MEL_FILTERS = _load_mel_filters()
    return _MEL_FILTERS


def audio_to_mel(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
) -> np.ndarray:
    """将 float32 单声道音频转为 log-mel spectrogram。

    Args:
        audio: float32 一维数组，范围 [-1, 1]
        sample_rate: 采样率 (默认 16000)
        n_mels: mel 滤波器组数量 (默认 80)
        n_fft: FFT 窗口大小 (默认 400 = 25ms @ 16kHz)
        hop_length: 帧移 (默认 160 = 10ms @ 16kHz)

    Returns:
        shape (n_mels, n_frames) 的 log-mel spectrogram
    """
    # STFT: 汉明窗 + FFT
    n_frames = (len(audio) - n_fft) // hop_length + 1
    if n_frames < 1:
        # 音频太短，零填充到至少一帧
        audio = np.pad(audio, (0, n_fft - len(audio)))
        n_frames = 1

    window = np.hamming(n_fft).astype(np.float32)
    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)

    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + n_fft] * window
        stft[:, i] = np.fft.rfft(frame)

    # 功率谱
    power = np.abs(stft) ** 2

    # Mel 滤波
    mel_filters = _get_mel_filters()
    # 截断滤波器到实际 FFT bin 数
    n_bins = power.shape[0]
    filters = mel_filters[:, :n_bins]

    mel_spec = filters @ power  # (n_mels, n_frames)

    # Log 压缩
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))

    # 归一化 (对齐 Whisper: log10 后 clip 到 [-2, 0] 再归一化)
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Tokenizer 封装 (openai-whisper)
# ═══════════════════════════════════════════════════════════════

# Whisper 特殊 token ID
WHISPER_TOKENS = {
    "SOT": 50257,           # Start of Transcript
    "EOT": 50256,           # End of Transcript
    "TRANSCRIBE": 50358,    # <|transcribe|>
    "TRANSLATE": 50357,     # <|translate|>
    "NO_TIMESTAMPS": 50362, # <|notimestamps|>
    "LANGUAGE_START": 50258, # first language token
    "LANGUAGE_END": 50356,   # last language token (exclusive)
    "TIMESTAMP_BEGIN": 50363, # <|0.00|>
}

# 语言代码 → token 映射
LANGUAGE_TOKEN_MAP = {
    "zh": 50299, "en": 50258, "ja": 50286, "ko": 50288,
    "auto": None,
}


class WhisperTokenizer:
    """Whisper tokenizer 轻量封装。

    从 openai-whisper 模型目录加载 tokenizer 文件，
    提供 encode/decode + 特殊 token 管理。
    """

    def __init__(self, model_dir: str):
        t0 = time.time()
        self._model_dir = Path(model_dir)

        # 尝试直接加载 tokenizer.json
        tokenizer_path = self._find_tokenizer()

        try:
            from whisper.tokenizer import get_tokenizer
            self._tokenizer = get_tokenizer(
                multilingual=True,
                language="zh",
                task="transcribe",
            )
            # 加载 tokenizer 的内部 tiktoken 编码器
            logger.info("[MODEL] Tokenizer 加载完成 (%.1fs)", time.time() - t0)
        except Exception as e:
            logger.warning("[MODEL] openai-whisper tokenizer 加载失败: %s, 尝试备选方案", e)
            self._tokenizer = self._load_from_files(tokenizer_path)

    def _find_tokenizer(self) -> Optional[Path]:
        """在模型目录中查找 tokenizer 文件"""
        # 优先: 本地模型目录下的 tokenizer.json
        candidates = [
            self._model_dir / "tokenizer.json",
            self._model_dir / "assets" / "tokenizer.json",
            self._model_dir / "multilingual" / "tokenizer.json",
        ]
        for p in candidates:
            if p.exists():
                return p

        # 备选: openai-whisper 包内的 assets
        try:
            import whisper
            whisper_dir = Path(whisper.__file__).parent / "assets"
            return whisper_dir / "tokenizer.json"
        except ImportError:
            pass

        return None

    def _load_from_files(self, path: Optional[Path]):
        """从文件加载 tokenizer (备选方案)"""
        import json

        if path is None or not path.exists():
            raise FileNotFoundError(
                "无法找到 Whisper tokenizer 文件。"
                "请安装: pip install openai-whisper"
            )

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 使用 tiktoken 加载
        try:
            import tiktoken
            self._encoding = tiktoken.Encoding(
                name="whisper",
                pat_str=data["pat_str"],
                mergeable_ranks=data["mergeable_ranks"],
                special_tokens=data.get("special_tokens", {}),
            )
            return self
        except ImportError:
            raise ImportError("需要 tiktoken: pip install tiktoken")

    @property
    def sot(self) -> int:
        return WHISPER_TOKENS["SOT"]

    @property
    def eot(self) -> int:
        return WHISPER_TOKENS["EOT"]

    def get_language_token(self, language: str) -> Optional[int]:
        return LANGUAGE_TOKEN_MAP.get(language)

    def get_task_token(self, task: str) -> int:
        if task == "translate":
            return WHISPER_TOKENS["TRANSLATE"]
        return WHISPER_TOKENS["TRANSCRIBE"]

    def get_initial_tokens(self, language: str, task: str = "transcribe") -> list[int]:
        """构建初始 token 序列: [SOT, language, task, no_timestamps]"""
        tokens = [self.sot]
        lang_token = self.get_language_token(language)
        if lang_token is not None:
            tokens.append(lang_token)
        tokens.append(self.get_task_token(task))
        tokens.append(WHISPER_TOKENS["NO_TIMESTAMPS"])
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """将 token ID 列表解码为文本"""
        try:
            return self._tokenizer.decode(token_ids)
        except AttributeError:
            # 备选: tiktoken encoding
            text = self._encoding.decode(token_ids)
            return text

    def encode(self, text: str) -> list[int]:
        try:
            return self._tokenizer.encode(text)
        except AttributeError:
            return self._encoding.encode(text)


# ═══════════════════════════════════════════════════════════════
# Triton gRPC 客户端封装
# ═══════════════════════════════════════════════════════════════

class TritonClient:
    """Triton Inference Server gRPC 异步客户端。

    特性:
    - 惰性连接 (首次调用时建立)
    - 自动重试 (瞬态 gRPC 错误)
    - asyncio.wait_for 超时保护
    - 健康检查
    """

    def __init__(
        self,
        url: str = "localhost:8001",
        model_name: str = "whisper_large_v3",
        model_version: str = "1",
        timeout: float = 5.0,
        max_retries: int = 3,
    ):
        self._url = url
        self._model_name = model_name
        self._model_version = model_version
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = None
        self._lock = asyncio.Lock()

    async def _ensure_client(self):
        """惰性初始化 gRPC 客户端"""
        if self._client is not None:
            return

        async with self._lock:
            if self._client is not None:
                return

            try:
                import tritonclient.grpc.aio as grpcclient
                self._client = grpcclient.InferenceServerClient(
                    url=self._url,
                    verbose=False,
                )
                # 检查连通性
                live = await asyncio.wait_for(
                    self._client.is_server_live(),
                    timeout=self._timeout,
                )
                if live:
                    logger.info("[MODEL] Triton 连接已建立: %s", self._url)
                else:
                    raise RuntimeError("Triton server is not live")
            except ImportError:
                raise ImportError(
                    "tritonclient[grpc] 未安装。"
                    "请运行: pip install tritonclient[grpc]"
                )
            except Exception as e:
                self._client = None
                logger.error("[MODEL] Triton 连接失败 (%s): %s", self._url, e)
                raise

    async def is_ready(self) -> bool:
        """检查 Triton server 和模型是否就绪"""
        try:
            await self._ensure_client()
            ready = await asyncio.wait_for(
                self._client.is_model_ready(
                    self._model_name,
                    self._model_version,
                ),
                timeout=self._timeout,
            )
            return ready
        except Exception as e:
            logger.warning("[MODEL] Triton 健康检查失败: %s", e)
            return False

    async def infer(
        self,
        model_component: str,
        inputs: dict[str, np.ndarray],
        outputs: list[str],
    ) -> dict[str, np.ndarray]:
        """执行一次 Triton 推理。

        Args:
            model_component: 模型组件名 ("encoder" 或 "decoder")
            inputs: {input_name: numpy_array}
            outputs: 输出名称列表

        Returns:
            {output_name: numpy_array}

        Raises:
            asyncio.TimeoutError: 推理超时
            RuntimeError: gRPC 错误 (自动重试后仍失败)
        """
        import tritonclient.grpc as grpcclient

        await self._ensure_client()

        model_name = f"{self._model_name}_{model_component}"
        logger.debug("[MODEL] Triton infer: %s, inputs=%s",
                     model_name, list(inputs.keys()))

        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                # 构建 Triton inputs
                triton_inputs = []
                for name, data in inputs.items():
                    triton_inputs.append(
                        grpcclient.InferInput(name, list(data.shape), "FP32" if data.dtype == np.float32 else "INT64")
                    )
                    triton_inputs[-1].set_data_from_numpy(data)

                triton_outputs = [
                    grpcclient.InferRequestedOutput(name) for name in outputs
                ]

                result = await asyncio.wait_for(
                    self._client.infer(
                        model_name=model_name,
                        model_version=self._model_version,
                        inputs=triton_inputs,
                        outputs=triton_outputs,
                    ),
                    timeout=self._timeout,
                )

                return {
                    name: result.as_numpy(name) for name in outputs
                }

            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError(
                    f"Triton infer {model_name} 超时 ({self._timeout}s)"
                )
                logger.warning("[MODEL] %s (attempt %d/%d)",
                             last_error, attempt + 1, self._max_retries + 1)

            except Exception as e:
                last_error = e
                logger.warning("[MODEL] Triton infer 失败 (attempt %d/%d): %s",
                             attempt + 1, self._max_retries + 1, e)
                if attempt < self._max_retries:
                    await asyncio.sleep(0.1 * (2 ** attempt))  # 指数退避

        raise RuntimeError(f"Triton infer 失败 (已重试 {self._max_retries} 次): {last_error}")

    async def close(self):
        """关闭 gRPC channel"""
        if self._client is not None:
            try:
                await self._client.close()
                logger.info("[MODEL] Triton 连接已关闭")
            except Exception as e:
                logger.warning("[MODEL] Triton 关闭异常: %s", e)
            finally:
                self._client = None


# ═══════════════════════════════════════════════════════════════
# TranscriberV2 — 异步转写引擎
# ═══════════════════════════════════════════════════════════════

class TranscriberV2:
    """基于 Triton Inference Server 的异步 Whisper 转写引擎。

    用法:
        transcriber = TranscriberV2(triton_url="localhost:8001")
        await transcriber.initialize()
        text = await transcriber.transcribe_chunk(audio, 16000)
        await transcriber.close()
    """

    def __init__(
        self,
        triton_url: str = "localhost:8001",
        model_name: str = "whisper_large_v3",
        model_version: str = "1",
        language: str = "auto",
        beam_size: int = 5,
        task: str = "transcribe",
        timeout: float = 5.0,
        max_retries: int = 3,
        download_root: str = "./models",
        hf_endpoint: str = "",
    ):
        self.triton_url = triton_url
        self.model_name = model_name
        self.model_version = model_version
        self.language = language
        self.beam_size = beam_size
        self.task = task
        self.timeout = timeout
        self.max_retries = max_retries
        self.download_root = download_root
        self.hf_endpoint = hf_endpoint

        # 组件 (惰性初始化)
        self._client: Optional[TritonClient] = None
        self._tokenizer: Optional[WhisperTokenizer] = None

        # 统计
        self._total_requests = 0
        self._total_time = 0.0

    # ─── 初始化 ────────────────────────────────────────────────

    async def initialize(self):
        """初始化 Triton 客户端和 tokenizer"""
        logger.info("[MODEL] ========== 初始化 TranscriberV2 ==========")
        logger.info("[MODEL]   Triton URL     = %s", self.triton_url)
        logger.info("[MODEL]   model_name     = %s", self.model_name)
        logger.info("[MODEL]   language       = %s", self.language)
        logger.info("[MODEL]   beam_size      = %s", self.beam_size)
        logger.info("[MODEL]   task           = %s", self.task)
        logger.info("[MODEL]   timeout        = %.1fs", self.timeout)
        logger.info("[MODEL]   max_retries    = %d", self.max_retries)

        # 初始化 Triton 客户端
        t0 = time.time()
        self._client = TritonClient(
            url=self.triton_url,
            model_name=self.model_name,
            model_version=self.model_version,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        try:
            await self._client._ensure_client()
            logger.info("[MODEL] ✅ Triton 客户端就绪 (%.1fs)", time.time() - t0)
        except Exception as e:
            logger.error("[MODEL] ⚠️  Triton 连接未建立: %s (将按需重连)", e)

        # 初始化 Tokenizer
        t0 = time.time()
        try:
            self._tokenizer = self._init_tokenizer()
            logger.info("[MODEL] ✅ Tokenizer 就绪 (%.1fs)", time.time() - t0)
        except Exception as e:
            logger.warning("[MODEL] ⚠️  Tokenizer 初始化失败: %s", e)
            self._tokenizer = None

        # 预热 mel filterbank
        _get_mel_filters()
        logger.info("[MODEL] ========== TranscriberV2 初始化完成 ==========")

    def _init_tokenizer(self) -> WhisperTokenizer:
        """初始化 tokenizer"""
        # 确定模型目录
        model_dir = self.download_root
        if self.model_name:
            model_dir = os.path.join(model_dir, f"models--{self.model_name.replace('/', '--')}")

        # 如果指定了 HF endpoint
        if self.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", self.hf_endpoint)

        try:
            return WhisperTokenizer(model_dir)
        except Exception:
            # 尝试 Whisper 包内置 tokenizer
            return WhisperTokenizer("")

    # ─── 健康检查 ──────────────────────────────────────────────

    async def health_check(self) -> dict:
        """健康检查，返回状态字典"""
        result = {
            "triton_connected": False,
            "encoder_ready": False,
            "decoder_ready": False,
            "tokenizer_ready": self._tokenizer is not None,
            "total_requests": self._total_requests,
            "avg_latency_ms": (self._total_time / max(self._total_requests, 1)) * 1000,
        }

        if self._client is not None:
            try:
                result["triton_connected"] = await self._client.is_ready()
            except Exception:
                pass

        return result

    # ─── 配置更新 ──────────────────────────────────────────────

    async def update_config(
        self,
        language: str = None,
        beam_size: int = None,
        task: str = None,
        timeout: float = None,
    ):
        """更新转写配置（不触发模型重载）"""
        logger.info("[MODEL] 检查配置更新...")
        changed = False

        if language is not None and language != self.language:
            logger.info("[MODEL]   language 变更: %s -> %s", self.language, language)
            self.language = language
            changed = True

        if beam_size is not None and beam_size != self.beam_size:
            logger.info("[MODEL]   beam_size 变更: %d -> %d", self.beam_size, beam_size)
            self.beam_size = beam_size
            changed = True

        if task is not None and task != self.task:
            logger.info("[MODEL]   task 变更: %s -> %s", self.task, task)
            self.task = task
            changed = True

        if timeout is not None and timeout != self.timeout:
            logger.info("[MODEL]   timeout 变更: %.1f -> %.1f", self.timeout, timeout)
            self.timeout = timeout
            if self._client is not None:
                self._client._timeout = timeout
            changed = True

        if not changed:
            logger.info("[MODEL] 配置未变更")
        else:
            logger.info("[MODEL] 配置已更新")

    # ─── 主转写方法 ────────────────────────────────────────────

    async def transcribe_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """转写一段音频数据，返回文本。

        这是主要的公开 API，与 v1 Transcriber.transcribe_chunk() 签名兼容。

        Args:
            audio: Int16 或 float32 的单声道音频
            sample_rate: 采样率 (默认 16000)

        Returns:
            转写文本字符串 (失败时返回空字符串)
        """
        t0 = time.time()
        self._total_requests += 1

        # 1. 音频归一化
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max

        # 单声道
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio_dur = len(audio) / sample_rate
        logger.info("[MODEL] 开始转写: %.2fs 音频, %d samples, sr=%d",
                    audio_dur, len(audio), sample_rate)

        try:
            # 2. 音频 → mel spectrogram
            t1 = time.time()
            mel = audio_to_mel(audio, sample_rate)
            mel = np.expand_dims(mel, axis=0)  # (1, 80, n_frames)
            mel_time = time.time() - t1
            logger.debug("[MODEL] mel 转换完成: shape=%s, %.1fms",
                         mel.shape, mel_time * 1000)

            # 3. Triton Encoder 推理
            t2 = time.time()
            try:
                encoder_result = await self._run_encoder(mel)
            except Exception as e:
                logger.error("[MODEL] Encoder 推理失败: %s", e)
                return ""
            encoder_time = time.time() - t2
            logger.debug("[MODEL] Encoder 完成: %.1fms", encoder_time * 1000)

            # 4. Triton Decoder 自回归循环
            t3 = time.time()
            try:
                token_ids = await self._run_decoder(encoder_result)
            except Exception as e:
                logger.error("[MODEL] Decoder 推理失败: %s", e)
                return ""
            decoder_time = time.time() - t3
            logger.debug("[MODEL] Decoder 完成: %d tokens, %.1fms",
                         len(token_ids), decoder_time * 1000)

            # 5. Token IDs → 文本
            text = self._decode_tokens(token_ids)

            total_time = time.time() - t0
            self._total_time += total_time
            rtf = total_time / max(audio_dur, 0.01)

            logger.info(
                "[MODEL] 转写完成 (%.1fs, RTF=%.2f): "
                "mel=%.0fms enc=%.0fms dec=%.0fms "
                "tokens=%d text[%d]=\"%s\"",
                total_time, rtf,
                mel_time * 1000, encoder_time * 1000, decoder_time * 1000,
                len(token_ids), len(text),
                text[:100] + ("..." if len(text) > 100 else ""),
            )

            return text

        except Exception as e:
            logger.error("[MODEL] 转写异常 (%.1fs): %s", time.time() - t0, e)
            import traceback
            logger.error("[MODEL] %s", traceback.format_exc())
            return ""

    # ─── Encoder ───────────────────────────────────────────────

    async def _run_encoder(self, mel: np.ndarray) -> dict[str, np.ndarray]:
        """调用 Triton Whisper Encoder。

        Args:
            mel: shape (1, 80, n_frames) log-mel spectrogram

        Returns:
            {"encoder_output": np.ndarray (1, n_frames//2, d_model)}
        """
        result = await self._client.infer(
            model_component="encoder",
            inputs={"mel": mel.astype(np.float32)},
            outputs=["encoder_output"],
        )
        return result

    # ─── Decoder 自回归循环 ────────────────────────────────────

    async def _run_decoder(
        self,
        encoder_result: dict[str, np.ndarray],
    ) -> list[int]:
        """自回归解码循环 (greedy decoding)。

        每步调用 Triton Decoder 获取下一个 token 的 logits，
        选择最高概率的 token，直到生成 EOT 或达到最大长度。

        Args:
            encoder_result: {"encoder_output": ndarray (1, n_frames, d_model)}

        Returns:
            token ID 列表 (包含 SOT 但不包含 EOT)
        """
        encoder_output = encoder_result["encoder_output"]
        encoder_output = np.asarray(encoder_output, dtype=np.float32)

        # 确定语言
        lang = None if self.language == "auto" else self.language
        if lang is None:
            # 自动检测：先用 greedy 探测语言
            lang = "zh"  # 默认中文

        # 初始 token 序列
        if self._tokenizer:
            tokens = self._tokenizer.get_initial_tokens(lang, self.task)
        else:
            # 硬编码 fallback
            sot = WHISPER_TOKENS["SOT"]
            lang_tok = LANGUAGE_TOKEN_MAP.get(lang, 50299)
            task_tok = WHISPER_TOKENS["TRANSCRIBE"]
            tokens = [sot, lang_tok, task_tok, WHISPER_TOKENS["NO_TIMESTAMPS"]]

        eot = WHISPER_TOKENS["EOT"]
        max_tokens = min(len(encoder_output[0]) + 100, 448)  # Whisper 最大 token 数

        # Greedy decoding loop
        for step in range(max_tokens - len(tokens)):
            token_array = np.array([tokens], dtype=np.int64)  # (1, seq_len)

            result = await self._client.infer(
                model_component="decoder",
                inputs={
                    "tokens": token_array,
                    "encoder_output": encoder_output,
                },
                outputs=["logits"],
            )

            logits = result["logits"]  # (1, seq_len, vocab_size)
            next_logits = logits[0, -1, :]  # 最后一个位置的 logits
            next_token = int(np.argmax(next_logits))

            tokens.append(next_token)

            if next_token == eot:
                break

        # 过滤掉特殊 token
        text_tokens = [
            t for t in tokens
            if t < WHISPER_TOKENS["LANGUAGE_START"]
            or t > WHISPER_TOKENS["EOT"]
        ]

        return text_tokens

    # ─── Token 解码 ────────────────────────────────────────────

    def _decode_tokens(self, token_ids: list[int]) -> str:
        """将 token ID 列表解码为文本"""
        if not token_ids:
            return ""

        if self._tokenizer is not None:
            try:
                # 使用 Whisper 的 decode (含特殊 token 处理)
                text = self._tokenizer.decode(token_ids)
                return text.strip()
            except Exception as e:
                logger.warning("[MODEL] Token 解码失败: %s, 返回 ID 列表", e)

        # 终极 fallback: 返回 token IDs 字符串
        return f"[{','.join(map(str, token_ids))}]"

    # ─── 清理 ─────────────────────────────────────────────────

    async def close(self):
        """关闭连接，释放资源"""
        logger.info("[MODEL] 关闭 TranscriberV2...")
        if self._client is not None:
            await self._client.close()
            self._client = None
        logger.info("[MODEL] TranscriberV2 已关闭")
