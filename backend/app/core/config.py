"""
配置管理 — YAML 加载/保存 + HF_ENDPOINT 环境变量设置

注意: 必须在 import faster_whisper 之前调用 _preload_hf_endpoint()，
否则 huggingface_hub 不会使用镜像。run.py 启动时会自动处理。
"""
import json
import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger("realtime_asr")

# config.yaml 相对于本文件的路径: app/core/config.py → backend/app/core/ → backend/
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = _CONFIG_DIR / "config.yaml"


def preload_hf_endpoint(config_path: Path = None):
    """
    预读 YAML 设置 HF_ENDPOINT 环境变量。

    必须在任何会导入 faster_whisper（通过 huggingface_hub）的模块之前调用。
    huggingface_hub 在 import 时读取一次 HF_ENDPOINT，之后再改就无效了。
    """
    path = config_path or CONFIG_PATH
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    endpoint = cfg.get("model", {}).get("hf_endpoint", "")
    if endpoint:
        os.environ.setdefault("HF_ENDPOINT", endpoint)
        logger.info("[CFG] HF_ENDPOINT 已设置为: %s", endpoint)


def load_config(config_path: Path = None) -> dict:
    """从 YAML 文件加载配置，同步更新 HF_ENDPOINT 环境变量"""
    path = config_path or CONFIG_PATH
    logger.info("[CFG] 加载配置文件: %s", path)

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    hf_endpoint = model_cfg.get("hf_endpoint", "")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        logger.info("[CFG] HF_ENDPOINT = %s", hf_endpoint)

    logger.info(
        "[CFG] model_path=%s, device=%s, compute_type=%s, download_root=%s, hf_endpoint=%s",
        model_cfg.get("model_path"),
        model_cfg.get("device"),
        model_cfg.get("compute_type"),
        model_cfg.get("download_root"),
        model_cfg.get("hf_endpoint", "(default)"),
    )
    logger.info(
        "[CFG] language=%s, buffer_threshold=%ss",
        config.get("transcription", {}).get("language"),
        config.get("transcription", {}).get("buffer_threshold"),
    )
    return config


def save_config(config: dict, config_path: Path = None) -> None:
    """保存配置到 YAML 文件"""
    path = config_path or CONFIG_PATH
    logger.info("[CFG] 保存配置到文件")
    logger.debug("[CFG] 新配置内容: %s", json.dumps(config, indent=2, ensure_ascii=False))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)
