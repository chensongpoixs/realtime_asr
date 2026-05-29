"""
WhisperWeb 后端入口 - FastAPI + WebSocket 实时语音转写服务
"""
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from transcriber import Transcriber
from audio_processor import extract_audio_to_numpy, validate_media_file

# ─── 日志配置 ────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("whisperweb")

# 降低第三方库日志级别
logging.getLogger("faster_whisper").setLevel(logging.INFO)
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# ─── 请求日志中间件 ──────────────────────────────────────────

class RequestLogMiddleware(BaseHTTPMiddleware):
    """记录每个 HTTP 请求的详细信息"""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        query = request.url.query

        logger.info(f"[REQ] {method} {path}{'?' + query if query else ''}  <-- {client_ip}")

        try:
            response = await call_next(request)
            elapsed_ms = (time.time() - start) * 1000
            logger.info(
                f"[RES] {method} {path}  --> {response.status_code}  ({elapsed_ms:.0f}ms)"
            )
            return response
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            logger.error(f"[ERR] {method} {path}  --> {e}  ({elapsed_ms:.0f}ms)")
            raise


app = FastAPI(title="WhisperWeb Backend", version="1.0.0")

app.add_middleware(RequestLogMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 全局状态 ────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "config.yaml"
_config: dict = {}
_transcriber: Optional[Transcriber] = None


def load_config() -> dict:
    global _config
    logger.info(f"[CFG] 加载配置文件: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f)
    model_cfg = _config.get("model", {})
    logger.info(
        f"[CFG] model_path={model_cfg.get('model_path')}, "
        f"device={model_cfg.get('device')}, "
        f"compute_type={model_cfg.get('compute_type')}, "
        f"download_root={model_cfg.get('download_root')}, "
        f"hf_endpoint={model_cfg.get('hf_endpoint', '(default)')}"
    )
    logger.info(
        f"[CFG] language={_config.get('transcription', {}).get('language')}, "
        f"buffer_threshold={_config.get('transcription', {}).get('buffer_threshold')}s"
    )
    return _config


def save_config(config: dict) -> None:
    global _config
    _config = config
    logger.info(f"[CFG] 保存配置到文件")
    logger.debug(f"[CFG] 新配置内容: {json.dumps(config, indent=2, ensure_ascii=False)}")
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)


def get_transcriber() -> Transcriber:
    global _transcriber
    if _transcriber is None:
        cfg = _config.get("model", {})
        model_path = cfg.get("model_path", "medium")
        device = cfg.get("device", "cpu")
        compute_type = cfg.get("compute_type", "int8")
        language = _config.get("transcription", {}).get("language", "auto")
        download_root = cfg.get("download_root", "./models")
        hf_endpoint = cfg.get("hf_endpoint", "")

        logger.info(f"[MODEL] 初始化转写器")
        logger.info(f"[MODEL]   model_path   = {model_path}")
        logger.info(f"[MODEL]   device       = {device}")
        logger.info(f"[MODEL]   compute      = {compute_type}")
        logger.info(f"[MODEL]   language     = {language}")
        logger.info(f"[MODEL]   download     = {download_root}")
        logger.info(f"[MODEL]   hf_endpoint  = {hf_endpoint or '(default)'}")

        _transcriber = Transcriber(
            model_path=model_path,
            device=device,
            compute_type=compute_type,
            language=language,
            download_root=download_root,
            hf_endpoint=hf_endpoint,
        )
        t0 = time.time()
        logger.info(f"[MODEL] 开始加载模型...")
        _transcriber.load_model()
        logger.info(f"[MODEL] 模型加载完成 ({time.time() - t0:.1f}s)")
    return _transcriber


# ─── 启动事件 ────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info("=" * 60)
    logger.info("[INIT] WhisperWeb Backend 启动中...")
    logger.info(f"[INIT] 启动时间: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    load_config()
    server = _config.get("server", {})
    logger.info(f"[INIT] 服务地址: http://{server.get('host', '0.0.0.0')}:{server.get('port', 8765)}")
    logger.info(f"[INIT] API 端点:")
    logger.info(f"[INIT]   GET  /api/health          - 健康检查")
    logger.info(f"[INIT]   GET  /api/config          - 获取配置")
    logger.info(f"[INIT]   POST /api/config          - 更新配置")
    logger.info(f"[INIT]   POST /api/transcribe/file - 文件批量转写")
    logger.info(f"[INIT]   WS   /ws/transcribe       - WebSocket实时转写")
    logger.info("=" * 60)


# ─── REST API ────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    logger.debug("[API] 健康检查")
    transcriber_loaded = _transcriber is not None and _transcriber._model is not None
    return {
        "status": "ok",
        "model_loaded": transcriber_loaded,
        "model_path": _config.get("model", {}).get("model_path", "unknown"),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/config")
async def get_config():
    logger.info("[API] GET /api/config - 返回当前配置")
    logger.debug(f"[API] 配置内容: model_path={_config.get('model', {}).get('model_path')}")
    return JSONResponse(_config)


@app.post("/api/config")
async def update_config(body: dict):
    logger.info(f"[API] POST /api/config - 更新配置")
    logger.info(f"[API] 请求体: {json.dumps(body, ensure_ascii=False)}")

    cfg = load_config()

    for section in ["model", "server", "transcription"]:
        if section in body:
            old_val = cfg.get(section, {})
            cfg[section].update(body[section])
            logger.info(f"[API]   [{section}] {old_val} -> {cfg[section]}")

    save_config(cfg)

    model_cfg = cfg.get("model", {})
    t = get_transcriber()
    logger.info(f"[API] 更新转写器配置...")
    t.update_config(
        model_path=model_cfg.get("model_path"),
        device=model_cfg.get("device"),
        compute_type=model_cfg.get("compute_type"),
        language=cfg.get("transcription", {}).get("language", "auto"),
        download_root=model_cfg.get("download_root"),
        hf_endpoint=model_cfg.get("hf_endpoint"),
    )
    logger.info(f"[API] 配置更新完成")

    return JSONResponse({"status": "ok", "config": cfg})


@app.post("/api/transcribe/file")
async def transcribe_file(file: UploadFile = File(...)):
    logger.info(f"[API] POST /api/transcribe/file - 文件转写")
    logger.info(f"[API]   文件名: {file.filename}")
    logger.info(f"[API]   文件大小: {file.size} bytes" if file.size else "[API]   文件大小: unknown")

    suffix = Path(file.filename).suffix or ".tmp"
    tmp_path = Path("/tmp") / f"whisperweb_{file.filename}{suffix}"
    try:
        content = await file.read()
        tmp_path.write_bytes(content)
        logger.info(f"[API]   临时文件: {tmp_path} ({len(content)} bytes)")

        valid, info = validate_media_file(str(tmp_path))
        if not valid:
            logger.error(f"[API]   文件验证失败: {info}")
            return JSONResponse({"error": info}, status_code=400)
        logger.info(f"[API]   文件验证通过: {info}")

        t0 = time.time()
        logger.info(f"[API]   开始提取音频...")
        audio = extract_audio_to_numpy(str(tmp_path))
        logger.info(f"[API]   音频提取完成: {audio.shape}, {len(audio) / 16000:.1f}s ({time.time() - t0:.1f}s)")

        transcriber = get_transcriber()
        t0 = time.time()
        logger.info(f"[API]   开始转写...")
        text = transcriber.transcribe_chunk(audio)
        logger.info(f"[API]   转写完成 ({time.time() - t0:.1f}s)")
        logger.info(f"[API]   结果: \"{text[:200]}{'...' if len(text) > 200 else ''}\"")

        return JSONResponse({"text": text})
    except Exception as e:
        logger.error(f"[API]   转写异常: {e}")
        logger.error(f"[API]   {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
            logger.debug(f"[API]   清理临时文件: {tmp_path}")


# ─── WebSocket 实时转写 ──────────────────────────────────────

@app.websocket("/ws/transcribe")
async def websocket_transcribe(ws: WebSocket):
    client = f"{ws.client.host}:{ws.client.port}" if ws.client else "unknown"
    logger.info(f"[WS] 新连接请求 <-- {client}")
    await ws.accept()
    logger.info(f"[WS] 连接已接受 <-- {client}")

    transcriber = get_transcriber()
    audio_buffer = np.array([], dtype=np.float32)
    target_sr = 16000
    buffer_threshold = _config.get("transcription", {}).get("buffer_threshold", 2.0)
    buffer_samples = int(target_sr * buffer_threshold)
    chunk_count = 0
    total_samples = 0

    logger.info(f"[WS] buffer_threshold={buffer_threshold}s, buffer_samples={buffer_samples}")

    await ws.send_json({"type": "status", "message": "就绪，等待音频数据..."})

    try:
        while True:
            msg = await ws.receive()

            if "bytes" in msg:
                raw = msg["bytes"]
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer = np.concatenate([audio_buffer, samples])
                total_samples += len(samples)
                chunk_count += 1

                logger.debug(
                    f"[WS] 收到音频chunk #{chunk_count}: {len(raw)} bytes = {len(samples)} samples, "
                    f"buffer={len(audio_buffer)}/{buffer_samples} samples"
                )

                if len(audio_buffer) >= buffer_samples:
                    buffer_dur = len(audio_buffer) / target_sr
                    logger.info(f"[WS] 缓冲区达到阈值 ({buffer_dur:.1f}s)，触发转写")
                    t0 = time.time()
                    text = transcriber.transcribe_chunk(audio_buffer, target_sr)
                    elapsed = time.time() - t0
                    audio_buffer = np.array([], dtype=np.float32)
                    if text:
                        logger.info(f"[WS] 转写结果 ({elapsed:.2f}s): \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
                        await ws.send_json({
                            "type": "transcription",
                            "text": text,
                            "partial": False,
                        })
                    else:
                        logger.info(f"[WS] 转写结果为空 ({elapsed:.2f}s)")

            elif "text" in msg:
                data = json.loads(msg["text"])
                msg_type = data.get("type", "")
                logger.info(f"[WS] 收到控制消息: type={msg_type}, data={json.dumps(data, ensure_ascii=False)}")

                if msg_type == "config":
                    target_sr = data.get("sample_rate", 16000)
                    buffer_threshold = data.get("buffer_threshold", buffer_threshold)
                    buffer_samples = int(target_sr * buffer_threshold)
                    logger.info(f"[WS] 音频参数更新: sr={target_sr}, threshold={buffer_threshold}s")
                    await ws.send_json({"type": "status", "message": "音频参数已更新"})

                elif msg_type == "end":
                    logger.info(f"[WS] 收到结束信号，处理剩余 {len(audio_buffer)}/{total_samples} samples")
                    if len(audio_buffer) > 0:
                        t0 = time.time()
                        text = transcriber.transcribe_chunk(audio_buffer, target_sr)
                        elapsed = time.time() - t0
                        if text:
                            logger.info(f"[WS] 最终转写 ({elapsed:.2f}s): \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
                            await ws.send_json({
                                "type": "transcription",
                                "text": text,
                                "partial": False,
                            })
                    logger.info(f"[WS] 转写完成: {chunk_count} chunks, {total_samples / target_sr:.1f}s 音频")
                    await ws.send_json({"type": "status", "message": "转写完成"})
                    break

    except WebSocketDisconnect:
        logger.info(f"[WS] 客户端断开连接 <-- {client}")
    except Exception as e:
        logger.error(f"[WS] 异常: {e}")
        logger.error(f"[WS] {traceback.format_exc()}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        logger.info(f"[WS] 连接关闭 <-- {client}, 共处理 {chunk_count} chunks")


# ─── 启动入口 ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    # 启动时也打印一份配置摘要
    cfg = load_config()
    server = cfg.get("server", {})
    host = server.get("host", "0.0.0.0")
    port = server.get("port", 8765)

    print("\n" + "=" * 60)
    print("  WhisperWeb Backend")
    print(f"  http://{host}:{port}")
    print(f"  日志级别: DEBUG")
    print("=" * 60 + "\n")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )
