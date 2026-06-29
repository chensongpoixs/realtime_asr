"""
转写接口 — 文件批量转写 (REST) + 实时转写 (WebSocket)
"""
import json
import logging
import time
import traceback
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.services.audio_processor import extract_audio_to_numpy, validate_media_file

# ─── REST 路由（挂载到 /api）───
rest_router = APIRouter()
# ─── WebSocket 路由（挂载到根路径，无 /api 前缀）───
ws_router = APIRouter()

logger = logging.getLogger("realtime_asr")


@rest_router.post("/transcribe/file")
async def transcribe_file(request: Request, file: UploadFile = File(...)):
    logger.info("[API] POST /api/transcribe/file - 文件转写")
    logger.info("[API]   文件名: %s", file.filename)
    logger.info("[API]   文件大小: %s bytes" % file.size if file.size else "[API]   文件大小: unknown")

    suffix = Path(file.filename).suffix or ".tmp"
    tmp_path = Path("/tmp") / f"realtime_asr_{file.filename}{suffix}"
    try:
        content = await file.read()
        tmp_path.write_bytes(content)
        logger.info("[API]   临时文件: %s (%d bytes)", tmp_path, len(content))

        valid, info = validate_media_file(str(tmp_path))
        if not valid:
            logger.error("[API]   文件验证失败: %s", info)
            return JSONResponse({"error": info}, status_code=400)
        logger.info("[API]   文件验证通过: %s", info)

        t0 = time.time()
        logger.info("[API]   开始提取音频...")
        audio = extract_audio_to_numpy(str(tmp_path))
        logger.info("[API]   音频提取完成: %s, %.1fs (%.1fs)",
                    audio.shape, len(audio) / 16000, time.time() - t0)

        transcriber = request.app.state.transcriber
        t0 = time.time()
        logger.info("[API]   开始转写...")
        text = transcriber.transcribe_chunk(audio)
        logger.info("[API]   转写完成 (%.1fs)", time.time() - t0)
        logger.info("[API]   结果: \"%s\"",
                    text[:200] + ("..." if len(text) > 200 else ""))

        return JSONResponse({"text": text})
    except Exception as e:
        logger.error("[API]   转写异常: %s", e)
        logger.error("[API]   %s", traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
            logger.debug("[API]   清理临时文件: %s", tmp_path)


@ws_router.websocket("/ws/transcribe")
async def websocket_transcribe(ws: WebSocket):
    client = f"{ws.client.host}:{ws.client.port}" if ws.client else "unknown"
    logger.info("[WS] 新连接请求 <-- %s", client)
    await ws.accept()
    logger.info("[WS] 连接已接受 <-- %s", client)

    app_state = ws.app.state
    transcriber = app_state.transcriber
    config = app_state.config

    audio_buffer = np.array([], dtype=np.float32)
    target_sr = 16000
    buffer_threshold = config.get("transcription", {}).get("buffer_threshold", 2.0)
    buffer_samples = int(target_sr * buffer_threshold)
    chunk_count = 0
    total_samples = 0

    logger.info("[WS] buffer_threshold=%ss, buffer_samples=%d", buffer_threshold, buffer_samples)

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
                    "[WS] 收到音频chunk #%d: %d bytes = %d samples, buffer=%d/%d samples",
                    chunk_count, len(raw), len(samples), len(audio_buffer), buffer_samples
                )

                if len(audio_buffer) >= buffer_samples:
                    buffer_dur = len(audio_buffer) / target_sr
                    logger.info("[WS] 缓冲区达到阈值 (%.1fs)，触发转写", buffer_dur)
                    t0 = time.time()
                    try:
                        text = transcriber.transcribe_chunk(audio_buffer, target_sr)
                        elapsed = time.time() - t0
                        audio_buffer = np.array([], dtype=np.float32)
                        if text:
                            logger.info("[WS] 转写结果 (%.2fs): \"%s\"",
                                        elapsed,
                                        text[:100] + ("..." if len(text) > 100 else ""))
                            await ws.send_json({
                                "type": "transcription",
                                "text": text,
                                "partial": False,
                            })
                        else:
                            logger.info("[WS] 转写结果为空 (%.2fs)", elapsed)
                    except Exception as trans_err:
                        elapsed = time.time() - t0
                        logger.error("[WS] 转写失败 (%.2fs): %s", elapsed, trans_err)
                        logger.error("[WS] %s", traceback.format_exc())
                        audio_buffer = np.array([], dtype=np.float32)
                        try:
                            await ws.send_json({"type": "error", "message": f"转写失败: {str(trans_err)}"})
                        except Exception:
                            pass

            elif "text" in msg:
                data = json.loads(msg["text"])
                msg_type = data.get("type", "")
                logger.info("[WS] 收到控制消息: type=%s, data=%s", msg_type,
                           json.dumps(data, ensure_ascii=False))

                if msg_type == "config":
                    target_sr = data.get("sample_rate", 16000)
                    buffer_threshold = data.get("buffer_threshold", buffer_threshold)
                    buffer_samples = int(target_sr * buffer_threshold)
                    logger.info("[WS] 音频参数更新: sr=%d, threshold=%ss", target_sr, buffer_threshold)
                    await ws.send_json({"type": "status", "message": "音频参数已更新"})

                elif msg_type == "end":
                    logger.info("[WS] 收到结束信号，处理剩余 %d/%d samples",
                               len(audio_buffer), total_samples)
                    if len(audio_buffer) > 0:
                        t0 = time.time()
                        try:
                            text = transcriber.transcribe_chunk(audio_buffer, target_sr)
                            elapsed = time.time() - t0
                            if text:
                                logger.info("[WS] 最终转写 (%.2fs): \"%s\"",
                                            elapsed,
                                            text[:100] + ("..." if len(text) > 100 else ""))
                                await ws.send_json({
                                    "type": "transcription",
                                    "text": text,
                                    "partial": False,
                                })
                        except Exception as trans_err:
                            elapsed = time.time() - t0
                            logger.error("[WS] 最终转写失败 (%.2fs): %s", elapsed, trans_err)
                            logger.error("[WS] %s", traceback.format_exc())
                            try:
                                await ws.send_json({"type": "error",
                                                    "message": f"最终转写失败: {str(trans_err)}"})
                            except Exception:
                                pass
                    logger.info("[WS] 转写完成: %d chunks, %.1fs 音频",
                               chunk_count, total_samples / target_sr)
                    await ws.send_json({"type": "status", "message": "转写完成"})
                    break

    except (WebSocketDisconnect, RuntimeError):
        # RuntimeError: "Cannot call receive once a disconnect message has been received."
        # Starlette 新版在客户端提前断开时抛出此异常
        logger.info("[WS] 客户端断开连接 <-- %s", client)
    except Exception as e:
        logger.error("[WS] 异常: %s", e)
        logger.error("[WS] %s", traceback.format_exc())
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        logger.info("[WS] 连接关闭 <-- %s, 共处理 %d chunks", client, chunk_count)
