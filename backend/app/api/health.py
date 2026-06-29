"""
健康检查接口
"""
import logging
from datetime import datetime

from fastapi import APIRouter, Request

router = APIRouter()
logger = logging.getLogger("realtime_asr")


@router.get("/health")
async def health(request: Request):
    logger.debug("[API] 健康检查")
    transcriber = request.app.state.transcriber
    config = request.app.state.config
    transcriber_loaded = transcriber is not None and transcriber._model is not None
    return {
        "status": "ok",
        "model_loaded": transcriber_loaded,
        "model_path": config.get("model", {}).get("model_path", "unknown"),
        "timestamp": datetime.now().isoformat(),
    }
