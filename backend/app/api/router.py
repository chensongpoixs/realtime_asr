"""
API 路由聚合

统一注册所有子路由到 /api 前缀下。
WebSocket 路由单独导出，由 app/main.py 直接挂载到根路径。
"""
from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.config import router as config_router
from app.api.transcribe import rest_router as transcribe_rest_router
from app.api.transcribe import ws_router as transcribe_ws_router

__all__ = ["api_router", "ws_router"]

# REST API 路由（挂载到 /api）
api_router = APIRouter(prefix="/api")
api_router.include_router(health_router)
api_router.include_router(config_router)
api_router.include_router(transcribe_rest_router)

# WebSocket 路由（挂载到根路径，由 main.py 直接 include）
ws_router = transcribe_ws_router
