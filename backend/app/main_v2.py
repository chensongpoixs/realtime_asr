"""
RealTime ASR v2.0 — FastAPI 应用 (Triton Inference Server)

负责:
- FastAPI app 创建 + 中间件注册
- TranscriberV2 初始化（Triton gRPC 连接）
- 生命周期管理（启动/关闭）
- 前端静态文件托管 + SPA 回退

与 v1 (app/main.py) 的关系:
- 复用 v1 共享模块: core/config, core/logger, api/health, api/config, utils/ssl_utils
- 使用 v2 组件: services/transcriber_v2, api/transcribe_v2
- 加载独立配置: config_v2.yaml
"""
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

# ═══ 最早阶段：设置 HF_ENDPOINT (tokenizer 加载可能需要) ═══
# run_v2.py 启动时已经预读 config_v2.yaml 设置了 HF_ENDPOINT，这里作为二次保障。
from app.core.config import preload_hf_endpoint
_CONFIG_PATH_V2 = Path(__file__).parent.parent / "config_v2.yaml"
preload_hf_endpoint(_CONFIG_PATH_V2)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logger import setup_logging

logger = logging.getLogger("realtime_asr.v2")


# ═══════════════════════════════════════════════════════════════
# 请求日志中间件 (与 v1 相同实现)
# ═══════════════════════════════════════════════════════════════

class RequestLogMiddleware(BaseHTTPMiddleware):
    """记录每个 HTTP 请求的详细信息"""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        query = request.url.query

        logger.info("[REQ] %s %s%s  <-- %s",
                    method, path, ("?" + query) if query else "", client_ip)

        try:
            response = await call_next(request)
            elapsed_ms = (time.time() - start) * 1000
            logger.info("[RES] %s %s  --> %s  (%.0fms)",
                        method, path, response.status_code, elapsed_ms)
            return response
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            logger.error("[ERR] %s %s  --> %s  (%.0fms)", method, path, e, elapsed_ms)
            raise


# ═══════════════════════════════════════════════════════════════
# 应用创建
# ═══════════════════════════════════════════════════════════════

app_v2 = FastAPI(title="RealTime ASR Backend", version="2.0.0")

app_v2.add_middleware(RequestLogMiddleware)
app_v2.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════
# 组件初始化
# ═══════════════════════════════════════════════════════════════

async def _init_transcriber_v2(config: dict):
    """初始化并连接 TranscriberV2 (Triton)"""
    from app.services.transcriber_v2 import TranscriberV2

    triton_cfg = config.get("triton", {})
    model_cfg = config.get("model", {})
    trans_cfg = config.get("transcription", {})

    triton_url = triton_cfg.get("url", "localhost:8001")
    model_name = triton_cfg.get("model_name", "whisper_large_v3")
    model_version = str(triton_cfg.get("model_version", "1"))
    language = trans_cfg.get("language", "auto")
    beam_size = trans_cfg.get("beam_size", 5)
    task = trans_cfg.get("task", "transcribe")
    timeout = triton_cfg.get("timeout", 5.0)
    max_retries = triton_cfg.get("max_retries", 3)
    download_root = model_cfg.get("download_root", "./models")
    hf_endpoint = model_cfg.get("hf_endpoint", "")

    logger.info("[INIT] ========== 初始化 TranscriberV2 (Triton) ==========")
    logger.info("[INIT]   Triton URL     = %s", triton_url)
    logger.info("[INIT]   model_name     = %s", model_name)
    logger.info("[INIT]   model_version  = %s", model_version)
    logger.info("[INIT]   language       = %s", language)
    logger.info("[INIT]   beam_size      = %d", beam_size)
    logger.info("[INIT]   task           = %s", task)
    logger.info("[INIT]   timeout        = %.1fs", timeout)
    logger.info("[INIT]   max_retries    = %d", max_retries)

    transcriber = TranscriberV2(
        triton_url=triton_url,
        model_name=model_name,
        model_version=model_version,
        language=language,
        beam_size=beam_size,
        task=task,
        timeout=timeout,
        max_retries=max_retries,
        download_root=download_root,
        hf_endpoint=hf_endpoint,
    )

    t0 = time.time()
    logger.info("[INIT] 正在初始化 TranscriberV2...")
    await transcriber.initialize()
    logger.info("[INIT] ✅ TranscriberV2 初始化完成 (%.1fs)", time.time() - t0)

    return transcriber


# ═══════════════════════════════════════════════════════════════
# 路由注册
# ═══════════════════════════════════════════════════════════════

# ─── v2 路由聚合 (复用 v1 的 health + config, 使用 v2 的 transcribe) ───
from fastapi import APIRouter
from app.api.health import router as health_router
from app.api.config import router as config_router
from app.api.transcribe_v2 import rest_router_v2, ws_router_v2

# 创建 v2 专用 API 路由 (不含 v1 的 /api/transcribe/file)
api_router_v2 = APIRouter(prefix="/api")
api_router_v2.include_router(health_router)
api_router_v2.include_router(config_router)
api_router_v2.include_router(rest_router_v2)

# 注册路由
app_v2.include_router(api_router_v2)         # /api/health, /api/config, /api/transcribe/file (v2)
app_v2.include_router(ws_router_v2)           # /ws/transcribe (v2 async)


# ═══════════════════════════════════════════════════════════════
# 生命周期
# ═══════════════════════════════════════════════════════════════

@app_v2.on_event("startup")
async def startup():
    logger.info("=" * 60)
    logger.info("[INIT] RealTime ASR Backend v2.0 (Triton) 启动中...")
    logger.info("[INIT] 启动时间: %s", datetime.now().isoformat())
    logger.info("=" * 60)

    # 日志
    setup_logging()

    # 加载 v2 配置
    from app.core.config import load_config
    config = load_config(str(_CONFIG_PATH_V2))
    app_v2.state.config = config

    server = config.get("server", {})
    use_ssl = server.get("ssl_enabled", False)
    proto = "https" if use_ssl else "http"
    logger.info("[INIT] 服务地址: %s://%s:%s",
                proto, server.get("host", "0.0.0.0"), server.get("port", 9766))

    triton_cfg = config.get("triton", {})
    logger.info("[INIT] Triton 服务器: %s", triton_cfg.get("url", "localhost:8001"))
    logger.info("[INIT] 模型名称: %s v%s",
                triton_cfg.get("model_name", "whisper_large_v3"),
                triton_cfg.get("model_version", "1"))

    logger.info("[INIT] API 端点:")
    logger.info("[INIT]   GET  /api/health          - 健康检查 (含 Triton 状态)")
    logger.info("[INIT]   GET  /api/config          - 获取配置")
    logger.info("[INIT]   POST /api/config          - 更新配置")
    logger.info("[INIT]   POST /api/transcribe/file - 文件批量转写")
    logger.info("[INIT]   WS   /ws/transcribe       - WebSocket 实时转写 (async)")

    frontend_cfg = config.get("frontend", {})
    dist_path = frontend_cfg.get("dist_path", "../frontend/dist")
    logger.info("[INIT] 前端文件: %s", Path(dist_path).resolve())

    logger.info("=" * 60)

    # 初始化 TranscriberV2
    logger.info("[INIT] 正在连接 Triton Inference Server...")
    try:
        app_v2.state.transcriber = await _init_transcriber_v2(config)

        # 健康检查
        health = await app_v2.state.transcriber.health_check()
        if health.get("triton_connected"):
            logger.info("[INIT] ✅ Triton 连接正常，模型就绪")
        else:
            logger.warning("[INIT] ⚠️  Triton 健康检查未通过: %s", health)
            logger.warning("[INIT]    服务仍会启动，但转写可能失败")
    except Exception as e:
        logger.error("[INIT] ❌ TranscriberV2 初始化失败: %s", e)
        logger.error("[INIT] %s", traceback.format_exc())
        logger.warning("[INIT] 服务仍会启动，转写将在首次请求时按需重连")

    # 静态文件托管
    if config.get("server", {}).get("serve_static", False):
        static_dir = os.path.abspath(config.get("frontend", {}).get("dist_path", "../frontend/dist"))
        if os.path.exists(static_dir):
            app_v2.mount(
                "/assets",
                StaticFiles(directory=os.path.join(static_dir, "assets")),
                name="assets_v2",
            )
            logger.info("[INIT] 静态文件托管: %s", static_dir)


@app_v2.on_event("shutdown")
async def shutdown():
    logger.info("[INIT] 服务关闭中...")
    if hasattr(app_v2.state, "transcriber") and app_v2.state.transcriber is not None:
        try:
            await app_v2.state.transcriber.close()
        except Exception as e:
            logger.warning("[INIT] TranscriberV2 关闭异常: %s", e)
    logger.info("[INIT] 服务已关闭")


# ═══════════════════════════════════════════════════════════════
# 根路由
# ═══════════════════════════════════════════════════════════════

@app_v2.get("/")
async def index():
    config = app_v2.state.config
    if config is None:
        return HTMLResponse("<h2>RealTime ASR v2.0 (Triton) API 已启动</h2>")

    frontend_cfg = config.get("frontend", {})
    dist_dir = Path(frontend_cfg.get("dist_path", "../frontend/dist"))
    if not dist_dir.is_absolute():
        dist_dir = (_CONFIG_PATH_V2.parent / dist_dir).resolve()

    index_path = dist_dir / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())

    return HTMLResponse(
        "<h2>RealTime ASR v2.0 (Triton) API 已启动</h2>"
        "<p>前端未构建，请运行: cd frontend && npm run build</p>"
    )


# ═══════════════════════════════════════════════════════════════
# SPA 回退中间件
# ═══════════════════════════════════════════════════════════════

@app_v2.middleware("http")
async def spa_fallback_middleware_v2(request: Request, call_next):
    """SPA 回退中间件 — 将未匹配的非 API GET 请求返回 index.html。"""
    response = await call_next(request)

    if response.status_code == 404 and request.method == "GET":
        path = request.url.path
        if not path.startswith("/api/") and not path.startswith("/ws"):
            config = getattr(app_v2.state, "config", None)
            if config is not None:
                frontend_cfg = config.get("frontend", {})
                dist_dir = Path(frontend_cfg.get("dist_path", "../frontend/dist"))
                if not dist_dir.is_absolute():
                    dist_dir = (_CONFIG_PATH_V2.parent / dist_dir).resolve()
                index_path = dist_dir / "index.html"
                if index_path.exists():
                    with open(index_path, "r", encoding="utf-8") as f:
                        return HTMLResponse(f.read())

    return response


# ═══════════════════════════════════════════════════════════════
# 前端静态文件服务 + SPA 回退
# ═══════════════════════════════════════════════════════════════

@app_v2.get("/{full_path:path}")
async def serve_frontend_v2(full_path: str):
    """提供前端静态文件，未知路径回退到 index.html"""
    if full_path.startswith("api/") or full_path.startswith("ws/"):
        return JSONResponse({"error": "Not found"}, status_code=404)

    config = app_v2.state.config
    if config is None:
        return JSONResponse({"error": "配置未加载"}, status_code=500)

    frontend_cfg = config.get("frontend", {})
    dist_dir = Path(frontend_cfg.get("dist_path", "../frontend/dist"))

    if not dist_dir.is_absolute():
        dist_dir = (_CONFIG_PATH_V2.parent / dist_dir).resolve()

    file_path = dist_dir / full_path

    # 安全检查：防止目录穿越
    try:
        file_path.resolve().relative_to(dist_dir)
    except ValueError:
        return JSONResponse({"error": "Forbidden"}, status_code=403)

    if file_path.is_file():
        return FileResponse(str(file_path))

    # SPA 回退
    index_path = dist_dir / "index.html"
    if index_path.is_file():
        return FileResponse(str(index_path))

    return JSONResponse(
        {"error": "前端文件未找到，请先构建: cd frontend && npm run build"},
        status_code=404,
    )
