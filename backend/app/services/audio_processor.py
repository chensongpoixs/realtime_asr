"""
音视频处理：利用 ffmpeg 提取音频流
"""
import logging
import subprocess
import time

import numpy as np
from pathlib import Path

logger = logging.getLogger("realtime_asr")


def extract_audio_to_numpy(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    从音频/视频文件中提取音频，转为 16kHz 单声道 float32 numpy 数组

    Args:
        file_path: 音频/视频文件路径
        target_sr: 目标采样率，默认 16000

    Returns:
        float32 numpy 数组（已归一化到 [-1, 1]）
    """
    file_path = str(Path(file_path).resolve())
    file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0

    logger.info("[AUDIO] 开始提取音频: %s (%.1f MB)", file_path, file_size / 1024 / 1024)

    cmd = [
        "ffmpeg",
        "-i", file_path,
        "-f", "s16le",       # 16-bit PCM 输出
        "-acodec", "pcm_s16le",
        "-ar", str(target_sr),
        "-ac", "1",           # 单声道
        "-loglevel", "error",
        "-"
    ]

    logger.debug("[AUDIO] ffmpeg 命令: %s", " ".join(cmd))

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error("[AUDIO] ffmpeg 提取失败: %s", e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e))
        raise

    elapsed = time.time() - t0
    audio = np.frombuffer(proc.stdout, dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0

    duration = len(audio) / target_sr
    logger.info("[AUDIO] 音频提取完成: %d samples, %.1fs, %.1f MB PCM, 耗时 %.1fs (%.1fx realtime)",
                len(audio), duration, proc.stdout.__len__() / 1024 / 1024,
                elapsed, duration / elapsed if elapsed > 0 else 0)

    return audio


def validate_media_file(file_path: str) -> tuple[bool, str]:
    """验证文件是否为有效的音视频文件

    Args:
        file_path: 文件路径

    Returns:
        (是否有效, 详细信息) 元组
    """
    file_path = str(Path(file_path).resolve())

    logger.info("[AUDIO] 验证媒体文件: %s", file_path)

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=format_name,duration",
        "-of", "default=noprint_wrappers=1",
        file_path
    ]

    logger.debug("[AUDIO] ffprobe 命令: %s", " ".join(cmd))

    try:
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        elapsed = time.time() - t0

        if result.returncode != 0:
            err_msg = result.stderr.strip() if result.stderr else "未知错误"
            logger.error("[AUDIO] 文件验证失败 (%.1fs): %s", elapsed, err_msg)
            return False, f"无法解析文件: {err_msg}"

        logger.info("[AUDIO] 文件验证通过 (%.1fs): %s", elapsed, result.stdout.strip())
        return True, result.stdout.strip()

    except subprocess.TimeoutExpired:
        logger.error("[AUDIO] 文件验证超时 (>30s): %s", file_path)
        return False, "解析文件超时"
    except FileNotFoundError:
        logger.error("[AUDIO] ffmpeg/ffprobe 未安装")
        return False, "未安装 ffmpeg/ffprobe"
