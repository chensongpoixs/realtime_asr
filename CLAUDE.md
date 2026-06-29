# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RealTime ASR is a real-time speech-to-text system: a browser-based UI streams audio (file or microphone) over WebSocket to a Python backend running faster-whisper, which returns transcription results in real time. Supports Chinese, English, Japanese, and Korean.

## Commands

```bash
# Backend
cd realtime_asr/backend
pip install -r requirements.txt
python run.py                      # ★ 启动服务 → :9765
python run.py --port 8080          # 指定端口

# Frontend (run from frontend/ directory)
cd realtime_asr/frontend
npm install
npm run dev        # Vite dev server on :5173 (proxies /ws and /api to backend)
npm run build      # Production build → frontend/dist/
npm run preview    # Preview production build
```

**Prerequisites:** ffmpeg must be installed and on PATH (used for audio extraction + validation). Python 3.10+, Node.js 18+.

**Production mode:** The backend serves `frontend/dist/` as static files. Build the frontend first, then access the app directly at `http://localhost:9765` — the Vite dev server is not needed.

## Architecture

### Data flow

```
Browser (FileUpload or mic) ──WebSocket──▶ backend/main.py
    │                                           │
    │  binary Int16 PCM chunks                  │ accumulates → transcriber.transcribe_chunk()
    │  JSON control messages ({type:"end"})     │
    │                                           │
    ◀── JSON {type:"transcription", text:"..."} ┘
```

### Backend (`backend/`)

工程化目录结构（参照 realtime_rag 模式）：

```
backend/
├── run.py                     # ★ 启动入口（替代 python main.py）
├── main.py                    # 向后兼容层：from app.main import app
├── config.yaml                # 全局配置文件
├── requirements.txt
├── app/
│   ├── main.py                # FastAPI app 创建 + 生命周期 + 组件初始化 + SPA 回退
│   ├── api/                   # ── 路由处理层 ──
│   │   ├── router.py          # 统一路由注册
│   │   ├── health.py          # GET  /api/health
│   │   ├── config.py          # GET/POST /api/config
│   │   └── transcribe.py      # POST /api/transcribe/file + WS /ws/transcribe
│   ├── core/                  # ── 核心配置层 ──
│   │   ├── config.py          # YAML 加载/保存 + HF_ENDPOINT 预设置
│   │   └── logger.py          # 日志初始化
│   ├── services/              # ── 业务逻辑层 ──
│   │   ├── transcriber.py     # Whisper 模型封装 (faster-whisper)
│   │   └── audio_processor.py # ffmpeg 音频提取 + 验证
│   └── utils/                 # ── 工具层 ──
│       └── ssl_utils.py       # SSL 自签名证书生成 + 本机 IP 检测
```

**核心模块说明：**
- **`run.py`** — 启动入口：预读 YAML 设置 HF_ENDPOINT → SSL 证书检查 → uvicorn 启动
- **`app/main.py`** — FastAPI app 创建、中间件（CORS + 请求日志）、生命周期、SPA 回退
- **`app/api/transcribe.py`** — WebSocket `/ws/transcribe` (二进制 PCM 累积→阈值触发转写) + REST 文件转写
- **`app/services/transcriber.py`** — `Transcriber` 类封装 faster_whisper，支持热重载（`update_config()` 检测参数变更）
- **`app/services/audio_processor.py`** — `extract_audio_to_numpy()` + `validate_media_file()`（subprocess 调用 ffmpeg/ffprobe）
- **`app/utils/ssl_utils.py`** — 自签名证书生成（含 LAN IP SAN，支持 iOS WSS）
- **`config.yaml`** — 模型路径/设备/语言/buffer阈值/SSL/前端路径等全部配置

**⚠️ 关键约束：** `HF_ENDPOINT` 必须在 `import faster_whisper` 之前设置。
`run.py` 在导入任何 app 模块前预读 YAML 并设置 `os.environ["HF_ENDPOINT"]`。

### Frontend (`frontend/`)

- **`App.vue`** — Root layout: sidebar (ConfigPanel) + main area (FileUpload + TranscriptionBox). Owns connection/processing status state, passes transcription results downward via template refs.
- **`FileUpload.vue`** — Dual-mode component: **file mode** (drag-and-drop or pick a file, decode with Web Audio API, chunk into 1s Int16 PCM segments, send over WebSocket) and **mic mode** (getUserMedia → AudioContext → ScriptProcessorNode capturing at device sample rate, resampling to 16kHz with linear interpolation, batching every 2s). Both modes create WebSocket connections directly — the `useWebSocket.js` composable exists but is **not currently used** by the components (they inline their own WebSocket handling).
- **`ConfigPanel.vue`** — Reads/writes backend config via `GET/POST /api/config`. Supports preset models (tiny through large-v3) and local path entry. On save, the backend hot-reloads the model if the path/device changed.
- **`TranscriptionBox.vue`** — Displays confirmed + partial transcription text, with copy/clear buttons and a blinking cursor during active sessions.
- **`useWebSocket.js`** — A general-purpose WebSocket composable with auto-reconnect (up to 5 attempts, 2s delay), message accumulation, and binary send helpers. Available but not wired into the current components.
- **`vite.config.js`** — Dev server proxies `/ws` → `wss://localhost:9765` and `/api` → `https://localhost:9765`. Note: the proxy targets use HTTPS/WSS regardless of whether the backend has SSL enabled, which may need adjustment when running without SSL.

### WebSocket protocol

| Direction | Format | Purpose |
|-----------|--------|---------|
| Client → Server | Binary (Int16 PCM) | 16kHz mono audio chunk |
| Client → Server | `{"type":"config","sample_rate":16000}` | Audio parameter negotiation |
| Client → Server | `{"type":"end"}` | Signals end of audio stream |
| Server → Client | `{"type":"transcription","text":"...","partial":false}` | Transcription result |
| Server → Client | `{"type":"status","message":"..."}` | Status updates |
| Server → Client | `{"type":"error","message":"..."}` | Error notification |

### Key behaviors

- **Buffer-threshold transcription:** The backend accumulates audio samples until `buffer_threshold` seconds (default 2s) are reached, then runs transcription on the entire buffer and clears it. This means results arrive in bursts, not word-by-word streaming.
- **SSL for mobile:** Mobile browsers require HTTPS/WSS for `getUserMedia`. The backend auto-generates a self-signed certificate with Subject Alternative Names for all detected LAN IPs, enabling phone access. Users must accept the cert warning on first visit.
- **Model hot-reload:** Changing model_path or device via the config panel triggers `Transcriber.update_config()`, which sets `_model = None` and reloads on next use. This means the next transcription request will block while the model loads.
- **vConsole:** The frontend unconditionally instantiates vConsole (`src/main.js:6`) for mobile debugging. Remove or condition it for production use.

### 日志系统

日志格式统一为 `HH:MM:SS | LEVEL | logger_name | message`，通过 `[TAG]` 前缀区分模块来源：

| 标签 | 含义 | 使用模块 |
|------|------|----------|
| `[BOOT]` | 启动过程 | `run.py` |
| `[INIT]` | 初始化和生命周期 | `app/main.py` (startup/shutdown) |
| `[CFG]` | 配置加载/保存 | `app/core/config.py`, `app/api/config.py` |
| `[REQ]` | HTTP 请求接收 | `app/main.py` (RequestLogMiddleware) |
| `[RES]` | HTTP 响应返回 | `app/main.py` (RequestLogMiddleware) |
| `[ERR]` | 请求处理异常 | `app/main.py` (RequestLogMiddleware) |
| `[API]` | REST API 操作 | `app/api/health.py`, `app/api/config.py`, `app/api/transcribe.py` |
| `[WS]` | WebSocket 连接/通信 | `app/api/transcribe.py` |
| `[MODEL]` | 模型加载/推理 | `app/services/transcriber.py` |
| `[AUDIO]` | 音频处理 (ffmpeg) | `app/services/audio_processor.py` |
| `[SSL]` | SSL 证书操作 | `app/utils/ssl_utils.py` |
| `[SPA]` | 前端 SPA 回退 | `app/main.py` |
| `[SEC]` | 安全检查 | `app/main.py` |

日志级别约定：
- `DEBUG` — 每个音频 chunk 接收、SPA 回退、静态文件请求等高频事件
- `INFO` — 连接建立/关闭、转写请求/完成、模型加载步骤、配置变更等关键操作
- `WARNING` — 配置缺失、SSL 证书类型错误等可恢复异常
- `ERROR` — 模型加载失败、转写异常、ffmpeg 执行失败等需要关注的错误

每次转写都会记录：音频时长、sample 数、检测语言、segment 数、耗时、实时率。

---

## v2.0 — Triton Inference Server 高并发版本

v2.0 是实时转写的高并发版本，使用 **Triton Inference Server + ONNX Runtime + TensorRT FP16** 替代 faster-whisper，通过异步非阻塞架构 + dynamic batching 支持 **80-100 路**并发流式转写。

### 架构对比

| 维度 | v1.0 | v2.0 |
|------|------|------|
| 推理引擎 | faster-whisper (CTranslate2) | Triton + ONNX Runtime + TensorRT FP16 |
| 调用方式 | 同步阻塞 (event loop 被卡死) | 异步非阻塞 (`asyncio.create_task`) |
| 批处理 | 无 | Dynamic batching (50ms 窗口, max 32 batch) |
| GPU 实例 | 1 个模型 | 2 个 GPU 实例 (轮询) |
| 最大并发 | 1-2 路 | 80-100 路 |
| 默认端口 | 9765 | 9766 |

### v2.0 新增文件

```
backend/
├── config_v2.yaml                        # v2 独立配置 (Triton URL, 端口 9766)
├── requirements_v2.txt                   # v2 额外依赖 (tritonclient[grpc])
├── run_v2.py                             # v2 启动入口
├── app/
│   ├── main_v2.py                        # FastAPI v2 应用工厂
│   ├── api/
│   │   └── transcribe_v2.py              # 异步非阻塞 WebSocket handler
│   └── services/
│       └── transcriber_v2.py             # Triton gRPC 异步转写引擎
├── scripts/
│   ├── export_whisper_to_onnx.py         # Whisper → ONNX 导出脚本
│   └── start_triton_server.sh            # Triton Server Docker 启动脚本
└── triton_model_repo/
    ├── mel_filters.npy                   # Mel filterbank (导出产物)
    ├── whisper_encoder/
    │   ├── config.pbtxt                  # Encoder Triton 配置
    │   └── 1/model.onnx                  # Encoder ONNX 模型
    └── whisper_decoder/
        ├── config.pbtxt                  # Decoder Triton 配置
        └── 1/model.onnx                  # Decoder ONNX 模型
```

### v2.0 命令

```bash
# ─── 一次性: 导出 ONNX 模型 ───
cd realtime_asr/backend
pip install torch torchaudio onnx onnxruntime-gpu openai-whisper
python scripts/export_whisper_to_onnx.py --model large-v3

# ─── 启动 Triton Inference Server ───
bash scripts/start_triton_server.sh

# ─── 安装 v2 依赖 ───
pip install -r requirements_v2.txt

# ─── 启动 v2 后端 ───
python run_v2.py                     # → :9766
python run_v2.py --port 8080         # 指定端口

# ─── v1 和 v2 可同时运行 ───
python run.py                        # v1.0 → :9765
python run_v2.py                     # v2.0 → :9766
```

### v2.0 关键技术点

- **Triton 通信**: `tritonclient.grpc.aio.InferenceServerClient` — HTTP/2 多路复用
- **音频预处理**: Python 端 NumPy FFT → log-mel spectrogram (80-bin)，发送 mel 到 Triton
- **非阻塞并发**: `asyncio.create_task()` 后台转写，WebSocket receive loop 永不阻塞
- **Encoder dynamic batching**: `max_queue_delay=50000μs` (50ms)，Triton 自动聚合请求
- **Decoder dynamic batching**: `max_queue_delay=10000μs` (10ms)，延迟更敏感
- **TensorRT FP16**: ~2x 加速，精度损失 <0.5%
- **版本共存**: v1 文件零改动，v2 全部新建文件

### 详细文档

完整的部署指南、性能调优、故障排查见: [`docs/TRITON_DEPLOYMENT.md`](docs/TRITON_DEPLOYMENT.md)
