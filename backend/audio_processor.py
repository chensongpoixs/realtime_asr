"""
音视频处理：利用 ffmpeg 提取音频流
"""
import subprocess
import tempfile
import numpy as np
from pathlib import Path


def extract_audio_to_numpy(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    从音频/视频文件中提取音频，转为 16kHz 单声道 float32 numpy 数组
    """
    file_path = str(Path(file_path).resolve())

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

    proc = subprocess.run(cmd, capture_output=True, check=True)
    audio = np.frombuffer(proc.stdout, dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0
    return audio


def validate_media_file(file_path: str) -> tuple[bool, str]:
    """验证文件是否为有效的音视频文件"""
    file_path = str(Path(file_path).resolve())
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=format_name,duration",
        "-of", "default=noprint_wrappers=1",
        file_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False, f"无法解析文件: {result.stderr.strip()}"
        return True, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "解析文件超时"
    except FileNotFoundError:
        return False, "未安装 ffmpeg/ffprobe"
