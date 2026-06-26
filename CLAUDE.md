# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RealTime ASR is a real-time speech-to-text system: a browser-based UI streams audio (file or microphone) over WebSocket to a Python backend running faster-whisper, which returns transcription results in real time. Supports Chinese, English, Japanese, and Korean.

## Commands

```bash
# Backend (run from repo root)
python backend/main.py                     # Start server on port 9765
pip install -r backend/requirements.txt    # Install Python dependencies

# Frontend (run from frontend/ directory)
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

- **`main.py`** — FastAPI app entry point. Registers middleware (CORS, request logging), REST routes (`/api/config`, `/api/health`, `/api/transcribe/file`), and the WebSocket endpoint `/ws/transcribe`. On startup loads `config.yaml`, shares a global `Transcriber` singleton. At the bottom of the file: uvicorn launch with optional SSL, and self-signed certificate generation (with SANs for LAN IPs) for mobile WSS access.
- **`transcriber.py`** — Wraps `faster_whisper.WhisperModel`. `Transcriber.transcribe_chunk(audio: np.float32, sr=16000) → str` handles dtype conversion, mono downmix, language selection, and VAD filtering. `update_config()` detects which params changed and reloads the model only when necessary.
- **`audio_processor.py`** — Shells out to ffmpeg: extracts 16kHz mono s16le PCM from any media file, returns `np.float32` normalized to [-1, 1]. Also provides `validate_media_file()` via ffprobe.
- **`config.yaml`** — All runtime configuration: model path/device/compute_type, language, buffer threshold, server host/port/SSL settings, frontend dist path, HuggingFace mirror endpoint.

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
