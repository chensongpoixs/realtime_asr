"""
RealTime ASR — FastAPI 应用

负责:
- FastAPI app 创建 + 中间件注册
- 组件初始化（Transcriber 预加载）
- 生命周期管理（启动/关闭）
- 前端静态文件托管 + SPA 回退
"""
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

# ═══ 最早阶段：必须设置 HF_ENDPOINT 后再 import 任何会触发 faster_whisper 的模块 ═══
# run.py 启动时已经预读 YAML 设置了 HF_ENDPOINT，这里作为二次保障。
from app.core.config import preload_hf_endpoint
preload_hf_endpoint()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import load_config, CONFIG_PATH
from app.core.logger import setup_logging
from app.api.router import api_router, ws_router

logger = logging.getLogger("realtime_asr")


# ─── 请求日志中间件 ──────────────────────────────────────────

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


# ─── 应用创建 ────────────────────────────────────────────────

app = FastAPI(title="RealTime ASR Backend", version="1.0.0")

app.add_middleware(RequestLogMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 API 路由
app.include_router(api_router)
# 注册 WebSocket 路由（挂载到根路径，不在 /api 下）
app.include_router(ws_router)


# ─── 组件初始化 ──────────────────────────────────────────────

def _init_transcriber(config: dict):
    """初始化并预加载 Transcriber"""
    from app.services.transcriber import Transcriber

    cfg = config.get("model", {})
    model_path = cfg.get("model_path", "medium")
    device = cfg.get("device", "cpu")
    compute_type = cfg.get("compute_type", "int8")
    language = config.get("transcription", {}).get("language", "auto")
    download_root = cfg.get("download_root", "./models")
    hf_endpoint = cfg.get("hf_endpoint", "")

    logger.info("[MODEL] 初始化转写器")
    logger.info("[MODEL]   model_path   = %s", model_path)
    logger.info("[MODEL]   device       = %s", device)
    logger.info("[MODEL]   compute      = %s", compute_type)
    logger.info("[MODEL]   language     = %s", language)
    logger.info("[MODEL]   download     = %s", download_root)
    logger.info("[MODEL]   hf_endpoint  = %s", hf_endpoint or "(default)")

    transcriber = Transcriber(
        model_path=model_path,
        device=device,
        compute_type=compute_type,
        language=language,
        download_root=download_root,
        hf_endpoint=hf_endpoint,
    )
    t0 = time.time()
    logger.info("[MODEL] 开始加载模型...")
    transcriber.load_model()
    logger.info("[MODEL] 模型加载完成 (%.1fs)", time.time() - t0)
    return transcriber


# ─── 生命周期 ────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info("=" * 60)
    logger.info("[INIT] RealTime ASR Backend 启动中...")
    logger.info("[INIT] 启动时间: %s", datetime.now().isoformat())
    logger.info("=" * 60)

    setup_logging()
    config = load_config()
    app.state.config = config

    server = config.get("server", {})
    use_ssl = server.get("ssl_enabled", False)
    proto = "https" if use_ssl else "http"
    logger.info("[INIT] 服务地址: %s://%s:%s",
                proto, server.get("host", "0.0.0.0"), server.get("port", 9765))
    if not use_ssl:
        logger.info("[INIT] SSL 未启用（ssl_enabled=false）")

    logger.info("[INIT] API 端点:")
    logger.info("[INIT]   GET  /api/health          - 健康检查")
    logger.info("[INIT]   GET  /api/config          - 获取配置")
    logger.info("[INIT]   POST /api/config          - 更新配置")
    logger.info("[INIT]   POST /api/transcribe/file - 文件批量转写")
    logger.info("[INIT]   WS   /ws/transcribe       - WebSocket实时转写")

    frontend_cfg = config.get("frontend", {})
    dist_path = frontend_cfg.get("dist_path", "../frontend/dist")
    logger.info("[INIT] 前端文件: %s", Path(dist_path).resolve())
    logger.info("=" * 60)

    # 预加载模型
    logger.info("[INIT] 预加载 Whisper 模型...")
    try:
        app.state.transcriber = _init_transcriber(config)
        logger.info("[INIT] ✅ 模型预加载完成")
    except Exception as e:
        logger.error("[INIT] ❌ 模型预加载失败: %s", e)
        logger.error("[INIT] %s", traceback.format_exc())
        logger.warning("[INIT] 模型将在首次请求时按需加载")

    # 静态文件托管
    if config.get("server", {}).get("serve_static", False):
        static_dir = os.path.abspath(config.get("frontend", {}).get("dist_path", "../frontend/dist"))
        if os.path.exists(static_dir):
            app.mount("/assets", StaticFiles(directory=os.path.join(static_dir, "assets")),
                     name="assets")
            logger.info("[INIT] 静态文件托管: %s", static_dir)


@app.on_event("shutdown")
async def shutdown():
    logger.info("[INIT] 服务关闭")


# ─── 根路由 ──────────────────────────────────────────────────

@app.get("/")
async def index():
    config = app.state.config
    if config is None:
        return HTMLResponse("<h2>RealTime ASR API 已启动</h2>")

    frontend_cfg = config.get("frontend", {})
    dist_dir = Path(frontend_cfg.get("dist_path", "../frontend/dist"))
    if not dist_dir.is_absolute():
        dist_dir = (CONFIG_PATH.parent / dist_dir).resolve()

    index_path = dist_dir / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())

    return HTMLResponse(
        "<h2>RealTime ASR API 已启动</h2>"
        "<p>前端未构建，请运行: cd frontend && npm run build</p>"
    )


# ─── SPA 回退中间件 ──────────────────────────────────────────

@app.middleware("http")
async def spa_fallback_middleware(request: Request, call_next):
    """
    SPA 回退中间件 — 将未匹配的非 API GET 请求返回 index.html。

    处理流程:
    1. 正常执行请求（API 路由 / 静态文件优先匹配）
    2. 如果响应是 404 且不是 /api/ 开头的 GET 请求 → 返回 index.html

    比 catch-all 路由更可靠:
    - middleware 在路由匹配 + 挂载点之后执行，不会拦截 /assets/* 静态文件
    - 只处理 404 响应（而非所有未匹配请求），API 404 仍返回 JSON
    """
    response = await call_next(request)

    if response.status_code == 404 and request.method == "GET":
        path = request.url.path
        if not path.startswith("/api/") and not path.startswith("/ws"):
            config = getattr(app.state, "config", None)
            if config is not None:
                frontend_cfg = config.get("frontend", {})
                dist_dir = Path(frontend_cfg.get("dist_path", "../frontend/dist"))
                if not dist_dir.is_absolute():
                    dist_dir = (CONFIG_PATH.parent / dist_dir).resolve()
                index_path = dist_dir / "index.html"
                if index_path.exists():
                    with open(index_path, "r", encoding="utf-8") as f:
                        return HTMLResponse(f.read())

    return response


# ─── 前端静态文件服务 + SPA 回退 ────────────────────────────────

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """提供前端静态文件，未知路径回退到 index.html（SPA路由支持）"""
    # API/WS 路径不应走到这里，但做安全兜底
    if full_path.startswith("api/") or full_path.startswith("ws/"):
        return JSONResponse({"error": "Not found"}, status_code=404)

    config = app.state.config
    if config is None:
        return JSONResponse({"error": "配置未加载"}, status_code=500)

    frontend_cfg = config.get("frontend", {})
    dist_dir = Path(frontend_cfg.get("dist_path", "../frontend/dist"))

    # 相对于 config.yaml 所在的 backend/ 目录
    if not dist_dir.is_absolute():
        dist_dir = (CONFIG_PATH.parent / dist_dir).resolve()

    file_path = dist_dir / full_path

    # 安全检查：防止目录穿越
    try:
        file_path.resolve().relative_to(dist_dir)
    except ValueError:
        return JSONResponse({"error": "Forbidden"}, status_code=403)

    if file_path.is_file():
        return FileResponse(str(file_path))

    # SPA 回退：任何不存在的路径返回 index.html
    index_path = dist_dir / "index.html"
    if index_path.is_file():
        return FileResponse(str(index_path))

    return JSONResponse(
        {"error": "前端文件未找到，请先构建: cd frontend && npm run build"},
        status_code=404,
    )
