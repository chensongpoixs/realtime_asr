"""
日志配置 — 统一日志格式和级别
"""
import logging
import sys


def setup_logging():
    """初始化全局日志配置"""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # 降低第三方库日志级别，减少噪音
    logging.getLogger("faster_whisper").setLevel(logging.INFO)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return logging.getLogger("realtime_asr")
