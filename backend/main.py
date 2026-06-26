"""
RealTime ASR 后端入口 - FastAPI + WebSocket 实时语音转写服务
"""
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

# ─── 日志配置（必须在最早进行，确保后续日志能输出）────────────

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("realtime_asr")

# 降低第三方库日志级别
logging.getLogger("faster_whisper").setLevel(logging.INFO)
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ─── 提前加载配置，设置 HF_ENDPOINT（必须在 import faster_whisper 之前）───
# huggingface_hub 在 import 时读取 HF_ENDPOINT 环境变量，
# 而 faster_whisper → huggingface_hub，所以必须在 import transcriber 之前设置。
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
    _preload_config = yaml.safe_load(_f)
_hf_endpoint = _preload_config.get("model", {}).get("hf_endpoint", "")
if _hf_endpoint:
    os.environ["HF_ENDPOINT"] = _hf_endpoint
    logger.info(f"[INIT] HF_ENDPOINT 已设置为: {_hf_endpoint}")

from transcriber import Transcriber
from audio_processor import extract_audio_to_numpy, validate_media_file


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


app = FastAPI(title="RealTime ASR Backend", version="1.0.0")

app.add_middleware(RequestLogMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 全局状态 ────────────────────────────────────────────────

_config: dict = {}
_transcriber: Optional[Transcriber] = None


def load_config() -> dict:
    global _config
    logger.info(f"[CFG] 加载配置文件: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f)
    model_cfg = _config.get("model", {})

    # 更新 HF_ENDPOINT 环境变量（配置可能在运行中被修改）
    _hf = model_cfg.get("hf_endpoint", "")
    if _hf:
        os.environ["HF_ENDPOINT"] = _hf
        logger.info(f"[CFG] HF_ENDPOINT = {_hf}")

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
    logger.info("[INIT] RealTime ASR Backend 启动中...")
    logger.info(f"[INIT] 启动时间: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    load_config()
    server = _config.get("server", {})
    use_ssl = server.get("ssl_enabled", False)
    proto = "https" if use_ssl else "http"
    logger.info(f"[INIT] 服务地址: {proto}://{server.get('host', '0.0.0.0')}:{server.get('port', 9765)}")
    if not use_ssl:
        logger.info(f"[INIT] SSL 未启用（ssl_enabled=false）")
    logger.info(f"[INIT] API 端点:")
    logger.info(f"[INIT]   GET  /api/health          - 健康检查")
    logger.info(f"[INIT]   GET  /api/config          - 获取配置")
    logger.info(f"[INIT]   POST /api/config          - 更新配置")
    logger.info(f"[INIT]   POST /api/transcribe/file - 文件批量转写")
    logger.info(f"[INIT]   WS   /ws/transcribe       - WebSocket实时转写")
    frontend_cfg = _config.get("frontend", {})
    dist_path = frontend_cfg.get("dist_path", "../frontend/dist")
    logger.info(f"[INIT] 前端文件: {Path(dist_path).resolve()}")
    logger.info("=" * 60)

    # 预加载模型（避免首次请求时才加载，耗时很长）
    logger.info("[INIT] 预加载 Whisper 模型...")
    try:
        get_transcriber()
        logger.info("[INIT] ✅ 模型预加载完成")
    except Exception as e:
        logger.error(f"[INIT] ❌ 模型预加载失败: {e}")
        logger.error(f"[INIT] {traceback.format_exc()}")
        logger.warning("[INIT] 模型将在首次请求时按需加载")


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

    for section in ["model", "server", "transcription", "frontend"]:
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
    tmp_path = Path("/tmp") / f"realtime_asr_{file.filename}{suffix}"
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
                    try:
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
                    except Exception as trans_err:
                        elapsed = time.time() - t0
                        logger.error(f"[WS] 转写失败 ({elapsed:.2f}s): {trans_err}")
                        logger.error(f"[WS] {traceback.format_exc()}")
                        audio_buffer = np.array([], dtype=np.float32)
                        try:
                            await ws.send_json({"type": "error", "message": f"转写失败: {str(trans_err)}"})
                        except Exception:
                            pass

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
                        try:
                            text = transcriber.transcribe_chunk(audio_buffer, target_sr)
                            elapsed = time.time() - t0
                            if text:
                                logger.info(f"[WS] 最终转写 ({elapsed:.2f}s): \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
                                await ws.send_json({
                                    "type": "transcription",
                                    "text": text,
                                    "partial": False,
                                })
                        except Exception as trans_err:
                            elapsed = time.time() - t0
                            logger.error(f"[WS] 最终转写失败 ({elapsed:.2f}s): {trans_err}")
                            logger.error(f"[WS] {traceback.format_exc()}")
                            try:
                                await ws.send_json({"type": "error", "message": f"最终转写失败: {str(trans_err)}"})
                            except Exception:
                                pass
                    logger.info(f"[WS] 转写完成: {chunk_count} chunks, {total_samples / target_sr:.1f}s 音频")
                    await ws.send_json({"type": "status", "message": "转写完成"})
                    break

    except (WebSocketDisconnect, RuntimeError):
        # RuntimeError: "Cannot call receive once a disconnect message has been received."
        # Starlette 新版在客户端提前断开时抛出此异常
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


# ─── 前端静态文件 & SPA回退 ─────────────────────────────────

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """提供前端静态文件，未知路径回退到 index.html（SPA路由支持）"""
    # API 路径不应走到这里，但做安全兜底
    if full_path.startswith("api/") or full_path.startswith("ws/"):
        return JSONResponse({"error": "Not found"}, status_code=404)

    frontend_cfg = _config.get("frontend", {})
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
        {"error": f"前端文件未找到，请先构建: cd frontend && npm run build"},
        status_code=404,
    )


# ─── SSL 证书自动生成 ────────────────────────────────────────


def get_local_ips() -> list[str]:
    """获取本机所有局域网 IP"""
    ips = ["127.0.0.1", "localhost"]
    try:
        hostname = socket.gethostname()
        ips.append(hostname)
        ips.append(f"{hostname}.local")
    except Exception:
        pass
    try:
        # 获取默认路由对应的 IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return list(set(ips))


def generate_self_signed_cert(cert_path: str, key_path: str) -> bool:
    """生成带 SAN 的自签名证书，解决 iOS WSS 证书主机名不匹配问题"""
    ips = get_local_ips()
    logger = logging.getLogger("realtime_asr")

    # 构建 SAN 扩展：DNS + IP
    san_entries = []
    for ip in ips:
        try:
            socket.inet_aton(ip)
            san_entries.append(f"IP:{ip}")
        except (socket.error, OSError):
            san_entries.append(f"DNS:{ip}")

    san_ext = ",".join(san_entries)
    logger.info(f"[SSL] 生成自签名证书, SAN: {san_ext}")

    # 创建临时配置文件（openssl 需要配置文件来指定 SAN）
    config = f"""
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = RealTime ASR

[v3_req]
subjectAltName = {san_ext}
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
"""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cnf", delete=False) as f:
            f.write(config)
            config_file = f.name

        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", key_path,
                "-out", cert_path,
                "-days", "3650",
                "-nodes",
                "-config", config_file,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        Path(config_file).unlink()
        logger.info(f"[SSL] 证书已生成: {cert_path}")
        logger.info(f"[SSL] 私钥已生成: {key_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[SSL] 证书生成失败: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("[SSL] 未安装 openssl，无法生成证书。请安装: apt install openssl")
        return False


# ─── 启动入口 ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    server = cfg.get("server", {})
    host = server.get("host", "0.0.0.0")
    port = server.get("port", 9765)
    use_ssl = server.get("ssl_enabled", False)

    ssl_cert = server.get("ssl_cert", "./fullchain.pem")
    ssl_key = server.get("ssl_key", "./privkey.pem")

    cert_path = Path(ssl_cert)
    key_path = Path(ssl_key)
    if not cert_path.is_absolute():
        cert_path = CONFIG_PATH.parent / cert_path
    if not key_path.is_absolute():
        key_path = CONFIG_PATH.parent / key_path

    # 证书总是生成（即使 SSL 未启用），方便用户下载安装后再切 HTTPS
    need_gen = False
    if not cert_path.exists() or not key_path.exists():
        print("[SSL] 证书文件不存在，自动生成...")
        need_gen = True
    else:
        try:
            first_line = cert_path.read_text(encoding="utf-8").strip().split("\n")[0]
            if "PRIVATE KEY" in first_line:
                print(f"[SSL] {ssl_cert} 是私钥不是证书，重新生成...")
                cert_path.unlink()
                key_path.unlink()
                need_gen = True
        except Exception:
            pass

    if need_gen:
        generate_self_signed_cert(str(cert_path), str(key_path))

    ssl_cert = str(cert_path.resolve())
    ssl_key = str(key_path.resolve())

    proto = "https" if use_ssl else "http"

    ips = get_local_ips()
    lan_ips = [ip for ip in ips if ip not in ("127.0.0.1", "localhost") and not ip.startswith("127.")]

    print("\n" + "=" * 60)
    print("  RealTime ASR Backend")
    print(f"  {proto}://{host}:{port}")
    if use_ssl:
        print(f"  SSL 证书: {ssl_cert}")
        for ip in lan_ips:
            print(f"  手机访问: https://{ip}:{port}")
    else:
        print(f"  协议: HTTP (SSL 未启用)")
    print(f"  日志级别: DEBUG")
    print("=" * 60 + "\n")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        ssl_certfile=ssl_cert if use_ssl else None,
        ssl_keyfile=ssl_key if use_ssl else None,
    )
