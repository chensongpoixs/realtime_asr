# RealTime ASR — 实时语音转文字

基于 **faster-whisper** (CTranslate2) 的实时语音转文字系统。浏览器端采集麦克风或上传文件，通过 WebSocket 实时传输音频流到 Python 后端进行转写，结果即时返回。

支持中文、英文、日文、韩语等多语言。

## 功能特性

- **实时流式转写** — WebSocket 双向通信，边说边出字
- **文件批量转写** — 上传音频/视频文件，一键转写（支持 mp3/wav/mp4/mkv 等格式）
- **模型热切换** — 前端面板切换模型（tiny ~ large-v3），无需重启服务
- **移动端适配** — 自动生成 SSL 自签名证书（含 LAN IP SAN），手机浏览器可访问
- **国内镜像加速** — 内置 HF-Mirror/ModelScope 模型下载源，无需科学上网
- **单文件部署** — 后端直接托管前端静态文件，无需 nginx

## 快速开始

### 前置依赖

- Python 3.10+
- Node.js 18+
- **ffmpeg** 必须在 PATH 中（音频提取和验证需要）

```bash
# 验证 ffmpeg 安装
ffmpeg -version
ffprobe -version
```

### 1. 安装后端

```bash
cd realtime_asr/backend
pip install -r requirements.txt
```

### 2. 配置模型

编辑 `backend/config.yaml`：

```yaml
model:
  model_path: large-v3       # tiny / base / small / medium / large-v3
  device: cuda               # cuda / cpu
  compute_type: float16      # int8 / float16 / float32
  download_root: ../../models/whisper
  hf_endpoint: https://hf-mirror.com   # 国内镜像
```

首次启动会自动下载模型到 `download_root` 目录。

### 3. 启动服务

```bash
# 推荐方式
python run.py

# 指定端口
python run.py --port 8080

# 兼容方式（旧）
python main.py
```

服务启动后：
- REST API: `http://localhost:9765`
- WebSocket: `ws://localhost:9765/ws/transcribe`
- API 文档: `http://localhost:9765/docs` (Swagger UI)

### 4. 启动前端（开发模式）

```bash
cd realtime_asr/frontend
npm install
npm run dev        # Vite 开发服务器 → http://localhost:5173
```

### 5. 生产部署

```bash
# 1. 构建前端
cd frontend && npm run build

# 2. 启动后端（自动托管前端静态文件）
cd ../backend && python run.py

# 3. 直接访问 http://localhost:9765
```

## 项目结构

```
realtime_asr/
├── README.md
├── CLAUDE.md                       # Claude Code 项目指南
├── DEVELOPMENT_PLAN.md             # 开发计划
│
├── backend/
│   ├── run.py                      # ★ 启动入口
│   ├── main.py                     # 向后兼容层
│   ├── config.yaml                 # 全局配置
│   ├── requirements.txt
│   ├── fullchain.pem / privkey.pem # SSL 证书（自动生成）
│   └── app/
│       ├── main.py                 # FastAPI 应用 + 中间件 + 生命周期
│       ├── api/                    # 路由处理层
│       │   ├── router.py           # 统一路由注册
│       │   ├── health.py           # GET  /api/health
│       │   ├── config.py           # GET/POST /api/config
│       │   └── transcribe.py       # POST /api/transcribe/file + WS /ws/transcribe
│       ├── core/                   # 核心配置层
│       │   ├── config.py           # YAML 加载/保存 + HF_ENDPOINT
│       │   └── logger.py           # 日志初始化
│       ├── services/               # 业务逻辑层
│       │   ├── transcriber.py      # Whisper 模型封装
│       │   └── audio_processor.py  # ffmpeg 音频处理
│       └── utils/                  # 工具层
│           └── ssl_utils.py        # SSL 证书生成 + IP 检测
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js              # Vite 配置 + dev proxy
    └── src/
        ├── main.js                 # Vue 入口
        ├── App.vue                 # 根组件
        ├── components/
        │   ├── ConfigPanel.vue     # 模型/设备/语言配置
        │   ├── FileUpload.vue      # 文件上传 + 麦克风录音
        │   └── TranscriptionBox.vue # 转写结果展示
        └── composables/
            └── useWebSocket.js     # WebSocket 连接管理
```

## 架构

```
浏览器 (Vue 3) ────WebSocket / REST──▶ Python FastAPI 后端 (:9765)
    │                                      │
    │  • 二进制 Int16 PCM 音频块            │ 累积 → 阈值触发
    │  • JSON 控制消息                     │ faster-whisper 转写
    │  • 文件上传 (multipart)               │
    │                                      │
    ◀── JSON 转写结果 / 状态消息 ────────────┘
```

### 数据流

```
[浏览器]                              [Python 后端]
   │                                      │
   │── WebSocket 连接 ────────────────────▶│
   │── 音频 chunk (binary Int16 PCM) ────▶│ 累积到 buffer
   │                                      │ 达到 buffer_threshold 秒
   │◀── {"type":"transcription","text":"..."} ──│ 转写 → 清空 buffer
   │                                      │
   │── {"type":"end"} ───────────────────▶│ 处理剩余音频
   │◀── 最终转写结果 ──────────────────────│
   │── 关闭连接                             │
```

## API 参考

### REST API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查：模型加载状态、模型路径、时间戳 |
| `GET` | `/api/config` | 获取当前完整配置（model/server/transcription/frontend） |
| `POST` | `/api/config` | 部分更新配置，模型参数变更自动热重载 |
| `POST` | `/api/transcribe/file` | 上传音视频文件，返回转写文本 |

### WebSocket 协议 — `/ws/transcribe`

**客户端 → 服务端：**

| 格式 | 说明 |
|------|------|
| Binary (Int16 PCM) | 16kHz 单声道音频块 |
| `{"type":"config","sample_rate":16000,"buffer_threshold":2}` | 音频参数协商 |
| `{"type":"end"}` | 音频流结束，触发最终转写 |

**服务端 → 客户端：**

| 格式 | 说明 |
|------|------|
| `{"type":"transcription","text":"...","partial":false}` | 转写结果 |
| `{"type":"status","message":"..."}` | 状态更新（就绪/完成） |
| `{"type":"error","message":"..."}` | 错误信息 |

## 配置参考

完整配置见 `backend/config.yaml`：

```yaml
frontend:
  dist_path: ../frontend/dist    # 生产构建输出路径

model:
  model_path: large-v3           # Whisper 模型大小或 HuggingFace 路径
  device: cuda                   # 推理设备: cuda / cpu
  compute_type: float16          # 计算精度: int8 / float16 / float32
  download_root: ../../models/whisper   # 模型下载目录
  hf_endpoint: https://hf-mirror.com   # HuggingFace 镜像（国内加速）

server:
  host: 0.0.0.0
  port: 9765
  ssl_enabled: true              # 启用 HTTPS/WSS（手机端必须）
  ssl_cert: ./fullchain.pem
  ssl_key: ./privkey.pem

transcription:
  buffer_threshold: 1            # 音频缓冲秒数（积累后触发转写）
  language: zh                   # 目标语言: zh/en/ja/ko/auto
  vad_filter: true               # 语音活动检测过滤静音
```

## 关键行为

- **缓冲阈值转写** — 默认积累 1 秒音频后批量转写（而非逐词流式），可通过面板调整
- **模型热重载** — 切换 model_path / device / compute_type 时自动卸载旧模型并加载新模型，下次请求生效
- **SSL 自动生成** — 移动端 `getUserMedia` 要求 HTTPS，后端启动时自动检测/生成自签名证书（包含所有 LAN IP 的 SAN）
- **预加载** — 启动时加载 Whisper 模型到内存，避免首次请求冷启动延迟（large-v3 约需 5-15 秒）
- **CORS 全开** — 开发阶段允许所有来源的跨域请求

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 后端框架 | FastAPI | 异步高性能，原生 WebSocket |
| STT 引擎 | faster-whisper | CTranslate2 加速，比原版快 4 倍 |
| 音频处理 | ffmpeg (subprocess) | 格式解码、重采样、声道转换 |
| 前端框架 | Vue 3 + Vite | 组合式 API，ES 模块打包 |
| 实时通信 | WebSocket | 二进制音频块 + JSON 控制信号 |
| UI 样式 | 原生 CSS | 零依赖，响应式布局 |
| 移动调试 | vConsole | 手机上查看日志和控制台 |

## 常见问题

**Q: 启动时提示 "No module named 'faster_whisper'"**
```bash
pip install faster-whisper>=1.0.0
```

**Q: ffmpeg 未找到**
- Windows: 下载 [ffmpeg](https://ffmpeg.org/download.html)，解压后将 `bin/` 加入 PATH
- macOS: `brew install ffmpeg`
- Linux: `apt install ffmpeg`

**Q: 手机无法访问**
1. 确保 `ssl_enabled: true`
2. 手机与电脑在同一局域网
3. 启动时控制台会打印手机访问 URL（如 `https://192.168.1.x:9765`）
4. 首次访问需信任自签名证书

**Q: CUDA out of memory**
- 调小模型：`model_path: medium` 或 `model_path: small`
- 或切换到 CPU：`device: cpu`

**Q: 模型下载慢**
设置 `hf_endpoint: https://hf-mirror.com` 使用国内镜像，首次下载 large-v3 约 2-5 分钟。

**Q: 如何添加自定义模型？**
将 `model_path` 设为本地路径即可，如 `model_path: /path/to/your/model`。
