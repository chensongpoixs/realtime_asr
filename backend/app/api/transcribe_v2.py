"""
转写接口 v2.0 — 异步非阻塞 WebSocket 实时转写 + REST 文件转写

与 v1 的关键区别:
- 使用 TranscriberV2 (Triton) 而非 Transcriber (faster-whisper)
- 所有转写调用通过 asyncio.create_task() 后台执行，不阻塞 receive loop
- 支持真正的多用户并发: 单 worker 可同时处理 80-100 路流

WebSocket 协议与 v1 完全兼容:
- Binary (Int16 PCM, 16kHz mono) → 音频数据
- JSON {type:"config", sample_rate:...} → 参数更新
- JSON {type:"end"} → 结束信号
- JSON {type:"transcription", text:"..."} → 转写结果
- JSON {type:"status"/"error", message:"..."} → 状态/错误
"""
import asyncio
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
rest_router_v2 = APIRouter()
# ─── WebSocket 路由（挂载到根路径，无 /api 前缀）───
ws_router_v2 = APIRouter()

logger = logging.getLogger("realtime_asr.v2")


# ═══════════════════════════════════════════════════════════════
# REST 文件转写
# ═══════════════════════════════════════════════════════════════

@rest_router_v2.post("/transcribe/file")
async def transcribe_file_v2(request: Request, file: UploadFile = File(...)):
    logger.info("[API] POST /api/transcribe/file - 文件转写 (v2)")
    logger.info("[API]   文件名: %s", file.filename)
    logger.info("[API]   文件大小: %s bytes" % file.size if file.size else "[API]   文件大小: unknown")

    suffix = Path(file.filename).suffix or ".tmp"
    tmp_path = Path("/tmp") / f"realtime_asr_v2_{file.filename}{suffix}"
    try:
        content = await file.read()
        tmp_path.write_bytes(content)
        logger.info("[API]   临时文件: %s (%d bytes)", tmp_path, len(content))

        valid, info = validate_media_file(str(tmp_path))
        if not valid:
            logger.error("[API]   文件验证失败: %s", info)
            return JSONResponse({"error": info}, status_code=400)

        t0 = time.time()
        audio = extract_audio_to_numpy(str(tmp_path))
        logger.info("[API]   音频提取完成: %s, %.1fs (%.1fs)",
                    audio.shape, len(audio) / 16000, time.time() - t0)

        transcriber = request.app.state.transcriber
        t0 = time.time()
        logger.info("[API]   开始转写 (Triton)...")
        text = await transcriber.transcribe_chunk(audio)
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


# ═══════════════════════════════════════════════════════════════
# WebSocket 实时转写
# ═══════════════════════════════════════════════════════════════

@ws_router_v2.websocket("/ws/transcribe")
async def websocket_transcribe_v2(ws: WebSocket):
    """WebSocket 实时转写端点 (v2 异步非阻塞版本)。

    核心改进: 使用 asyncio.create_task() 将转写任务放到后台执行，
    WebSocket receive loop 永不阻塞，实现真正的多用户并发。
    """
    client = f"{ws.client.host}:{ws.client.port}" if ws.client else "unknown"
    logger.info("[WS] 新连接请求 <-- %s", client)
    await ws.accept()
    logger.info("[WS] 连接已接受 <-- %s", client)

    app_state = ws.app.state
    transcriber = app_state.transcriber
    config = app_state.config

    audio_buffer = np.array([], dtype=np.float32)
    target_sr = 16000
    buffer_threshold = config.get("transcription", {}).get("buffer_threshold", 1.0)
    buffer_samples = int(target_sr * buffer_threshold)
    chunk_count = 0
    total_samples = 0

    # 后台转写任务追踪
    pending_tasks: list[asyncio.Task] = []

    logger.info("[WS] buffer_threshold=%ss, buffer_samples=%d (v2 async)",
                buffer_threshold, buffer_samples)

    await ws.send_json({"type": "status", "message": "就绪 (Triton v2.0)，等待音频数据..."})

    # ─── 后台转写协程 ────────────────────────────────────────

    async def transcribe_and_send(
        audio: np.ndarray,
        sr: int,
        partial: bool = False,
    ):
        """后台执行转写并发送结果。"""
        t0 = time.time()
        try:
            text = await transcriber.transcribe_chunk(audio, sr)
            elapsed = time.time() - t0
            if text:
                # 安全发送 (WebSocket 并发写保护)
                logger.info("[WS] 后台转写完成 (%.2fs): \"%s\"",
                            elapsed,
                            text[:100] + ("..." if len(text) > 100 else ""))
                try:
                    await ws.send_json({
                        "type": "transcription",
                        "text": text,
                        "partial": partial,
                    })
                except RuntimeError as e:
                    # WebSocket 已关闭，正常情况
                    logger.debug("[WS] 发送跳过 (连接已关闭): %s", e)
            else:
                logger.info("[WS] 后台转写结果为空 (%.2fs)", elapsed)
        except Exception as e:
            elapsed = time.time() - t0
            logger.error("[WS] 后台转写失败 (%.2fs): %s", elapsed, e)
            logger.error("[WS] %s", traceback.format_exc())
            try:
                await ws.send_json({
                    "type": "error",
                    "message": f"转写失败: {str(e)}",
                })
            except Exception:
                pass

    # ─── 主循环 ────────────────────────────────────────────

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
                    chunk_count, len(raw), len(samples),
                    len(audio_buffer), buffer_samples,
                )

                if len(audio_buffer) >= buffer_samples:
                    buffer_dur = len(audio_buffer) / target_sr
                    logger.info("[WS] 缓冲区达到阈值 (%.1fs)，触发后台转写", buffer_dur)

                    # ★ 关键: 使用 asyncio.create_task() 后台执行，不阻塞 receive loop
                    task = asyncio.create_task(
                        transcribe_and_send(audio_buffer.copy(), target_sr)
                    )
                    pending_tasks.append(task)

                    # 清理已完成的任务 (避免列表无限增长)
                    pending_tasks = [t for t in pending_tasks if not t.done()]

                    # 清空缓冲区，继续接收下一批音频
                    audio_buffer = np.array([], dtype=np.float32)

            elif "text" in msg:
                data = json.loads(msg["text"])
                msg_type = data.get("type", "")
                logger.info("[WS] 收到控制消息: type=%s, data=%s",
                           msg_type, json.dumps(data, ensure_ascii=False))

                if msg_type == "config":
                    target_sr = data.get("sample_rate", 16000)
                    buffer_threshold = data.get("buffer_threshold", buffer_threshold)
                    buffer_samples = int(target_sr * buffer_threshold)
                    logger.info("[WS] 音频参数更新: sr=%d, threshold=%ss",
                               target_sr, buffer_threshold)
                    await ws.send_json({"type": "status", "message": "音频参数已更新"})

                elif msg_type == "end":
                    logger.info("[WS] 收到结束信号, buffer=%d/%d samples, pending=%d tasks",
                               len(audio_buffer), total_samples, len(pending_tasks))

                    # 1. 处理剩余缓冲区
                    if len(audio_buffer) > 0:
                        logger.info("[WS] 处理剩余 %d samples (%.2fs)",
                                   len(audio_buffer),
                                   len(audio_buffer) / target_sr)
                        task = asyncio.create_task(
                            transcribe_and_send(audio_buffer.copy(), target_sr, partial=False)
                        )
                        pending_tasks.append(task)
                        audio_buffer = np.array([], dtype=np.float32)

                    # 2. 等待所有后台转写任务完成
                    if pending_tasks:
                        logger.info("[WS] 等待 %d 个后台任务完成...", len(pending_tasks))
                        await asyncio.gather(*pending_tasks, return_exceptions=True)
                        logger.info("[WS] 所有后台任务已完成")

                    logger.info("[WS] 转写完成: %d chunks, %.1fs 音频",
                               chunk_count, total_samples / target_sr)
                    await ws.send_json({"type": "status", "message": "转写完成"})
                    break

    except (WebSocketDisconnect, RuntimeError):
        # RuntimeError: Starlette 在客户端提前断开时抛出
        logger.info("[WS] 客户端断开连接 <-- %s", client)
    except Exception as e:
        logger.error("[WS] 异常: %s", e)
        logger.error("[WS] %s", traceback.format_exc())
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # 取消所有未完成的后台任务
        for task in pending_tasks:
            if not task.done():
                task.cancel()
        logger.info("[WS] 连接关闭 <-- %s, 共处理 %d chunks", client, chunk_count)
