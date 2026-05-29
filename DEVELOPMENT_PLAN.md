# WhisperWeb 开发文档

## 项目概述

基于 OpenAI Whisper 模型的实时语音转文字系统，支持音频/视频输入，前后端分离架构。

## 技术选型

| 层级 | 技术 | 说明 |
|------|------|------|
| 后端框架 | Python FastAPI | 异步高性能，原生 WebSocket 支持 |
| STT 引擎 | faster-whisper | CTranslate2 加速，比原版 Whisper 快 4 倍 |
| 音视频处理 | ffmpeg-python | 提取音频流、格式转换 |
| 前端框架 | Vue 3 + Vite | 组合式 API，现代化构建工具 |
| 实时通信 | WebSocket | 双向流式传输音频数据和转写结果 |
| UI 样式 | 原生 CSS | 零依赖，简洁高效 |

## 项目目录结构

```
whisperweb/
├── DEVELOPMENT_PLAN.md       # 本开发文档
├── backend/
│   ├── requirements.txt      # Python 依赖
│   ├── config.yaml           # 模型路径等配置文件
│   ├── main.py               # FastAPI 入口，WebSocket/HTTP 路由
│   ├── transcriber.py        # Whisper 模型封装，转写逻辑
│   └── audio_processor.py    # 音视频处理（ffmpeg 提取音频）
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.js           # Vue 入口
│       ├── App.vue           # 根组件 + 整体布局
│       ├── components/
│       │   ├── ConfigPanel.vue         # 模型路径配置面板
│       │   ├── FileUpload.vue          # 文件上传组件
│       │   └── TranscriptionBox.vue    # 实时转写结果展示
│       └── composables/
│           └── useWebSocket.js         # WebSocket 连接管理
```

---

## 开发步骤

### 步骤 1：后端基础搭建 ✅
- [x] 创建 `backend/requirements.txt`
- [x] 创建 `backend/config.yaml` - 模型路径/设备/计算类型等配置
- [x] 实现 `backend/transcriber.py` - Whisper 模型加载、音频段转写
- [x] 实现 `backend/audio_processor.py` - 音视频文件音频提取
- [x] 实现 `backend/main.py` - FastAPI 应用、WebSocket 端点、REST API

### 步骤 2：前端基础搭建 ✅
- [x] 创建 `frontend/package.json`
- [x] 创建 `frontend/vite.config.js`
- [x] 创建 `frontend/index.html`
- [x] 创建 `frontend/src/main.js`

### 步骤 3：前端核心组件 ✅
- [x] 实现 `useWebSocket.js` - WebSocket 连接/重连/消息管理
- [x] 实现 `ConfigPanel.vue` - 模型路径配置（加载/保存）
- [x] 实现 `FileUpload.vue` - 文件选择、音频流式发送
- [x] 实现 `TranscriptionBox.vue` - 实时显示转写文字
- [x] 实现 `App.vue` - 组装所有组件

### 步骤 4：联调测试 ✅
- [x] 安装所有依赖（npm + pip）
- [x] 后端 Python 语法验证通过
- [x] 前端 Vite build 构建通过
- [x] 环境检查（ffmpeg 已就绪）
- [ ] 启动后端服务 `python backend/main.py`
- [ ] 启动前端 `npm run dev`（frontend 目录）
- [ ] 浏览器打开 http://localhost:5173 测试

---

## 数据流设计

```
[浏览器]                                [Python 后端]
   │                                        │
   │  1. 用户选择音频/视频文件                │
   │  2. 前端读取文件为 ArrayBuffer          │
   │                                        │
   │── WebSocket 连接 ──────────────────────▶│
   │                                        │
   │── 发送音频 chunk (binary) ────────────▶│
   │                                        │── 累积音频数据
   │                                        │── 达到阈值后调用 Whisper 转写
   │◀── 返回转写文本 (json) ────────────────│
   │                                        │
   │  3. 实时显示转写结果                     │
   │                                        │
   │── 发送结束信号 ────────────────────────▶│
   │                                        │── 处理剩余音频
   │◀── 返回最终转写结果 ───────────────────│
   │                                        │
   │── 关闭连接                              │
```

## WebSocket 协议

### 客户端 → 服务端

| 消息类型 | 格式 | 说明 |
|---------|------|------|
| 音频数据 | Binary (Int16 PCM) | 16kHz 单声道 PCM 音频块 |
| 控制消息 | JSON `{"type": "config", "sample_rate": 16000}` | 音频参数配置 |
| 结束信号 | JSON `{"type": "end"}` | 通知服务端音频发送完毕 |

### 服务端 → 客户端

| 消息类型 | 格式 | 说明 |
|---------|------|------|
| 转写结果 | JSON `{"type": "transcription", "text": "...", "partial": true/false}` | 实时/最终转写文本 |
| 状态消息 | JSON `{"type": "status", "message": "..."}` | 模型加载状态等 |
| 错误消息 | JSON `{"type": "error", "message": "..."}` | 错误信息 |

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/config` | 获取当前配置 |
| POST | `/api/config` | 更新配置（模型路径等） |
| GET | `/api/health` | 健康检查 |
| WS | `/ws/transcribe` | WebSocket 实时转写 |
| POST | `/api/transcribe/file` | HTTP 文件上传批量转写（备用） |
