"""
配置管理接口
"""
import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.config import load_config, save_config

router = APIRouter()
logger = logging.getLogger("realtime_asr")


@router.get("/config")
async def get_config(request: Request):
    logger.info("[API] GET /api/config - 返回当前配置")
    config = request.app.state.config
    logger.debug("[API] 配置内容: model_path=%s", config.get("model", {}).get("model_path"))
    return JSONResponse(config)


@router.post("/config")
async def update_config(request: Request, body: dict):
    logger.info("[API] POST /api/config - 更新配置")
    logger.info("[API] 请求体: %s", json.dumps(body, ensure_ascii=False))

    config = load_config()

    for section in ["model", "server", "transcription", "frontend"]:
        if section in body:
            old_val = config.get(section, {})
            config[section].update(body[section])
            logger.info("[API]   [%s] %s -> %s", section, old_val, config[section])

    save_config(config)
    request.app.state.config = config

    model_cfg = config.get("model", {})
    transcriber = request.app.state.transcriber
    logger.info("[API] 更新转写器配置...")
    transcriber.update_config(
        model_path=model_cfg.get("model_path"),
        device=model_cfg.get("device"),
        compute_type=model_cfg.get("compute_type"),
        language=config.get("transcription", {}).get("language", "auto"),
        download_root=model_cfg.get("download_root"),
        hf_endpoint=model_cfg.get("hf_endpoint"),
    )
    logger.info("[API] 配置更新完成")

    return JSONResponse({"status": "ok", "config": config})
