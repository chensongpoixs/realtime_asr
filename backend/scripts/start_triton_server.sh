#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 启动 Triton Inference Server (Whisper ONNX 模型)
# ═══════════════════════════════════════════════════════════════
#
# 前置条件:
#   1. Docker + NVIDIA Container Toolkit 已安装
#   2. ONNX 模型已导出: python scripts/export_whisper_to_onnx.py
#   3. triton_model_repo/ 目录存在且包含 encoder/decoder config
#
# 用法:
#   bash scripts/start_triton_server.sh
#   bash scripts/start_triton_server.sh --gpu 0         # 指定 GPU
#   bash scripts/start_triton_server.sh --port 8001      # 指定 gRPC 端口
#
# 验证:
#   curl http://localhost:8000/v2/health/ready
#   curl http://localhost:8000/v2/models/whisper_encoder/ready
# ═══════════════════════════════════════════════════════════════

set -e

# ─── 默认配置 ──────────────────────────────────────────────
TRITON_IMAGE="nvcr.io/nvidia/tritonserver:24.01-py3"
CONTAINER_NAME="triton_whisper"
GPU_ID="${GPU_ID:-all}"
GRPC_PORT="${GRPC_PORT:-8001}"
HTTP_PORT="${HTTP_PORT:-8000}"
METRICS_PORT="${METRICS_PORT:-8002}"
LOG_LEVEL="${LOG_LEVEL:-1}"  # 0=MIN, 1=INFO, 2=VERBOSE

# ─── 解析参数 ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --port)
            GRPC_PORT="$2"
            HTTP_PORT="$((GRPC_PORT - 1))"
            METRICS_PORT="$((GRPC_PORT + 1))"
            shift 2
            ;;
        --image)
            TRITON_IMAGE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--gpu GPU_ID] [--port GRPC_PORT] [--image IMAGE] [--log-level 0|1|2]"
            exit 1
            ;;
    esac
done

# ─── 模型仓库路径 ──────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_REPO="$(cd "$SCRIPT_DIR/../triton_model_repo" && pwd)"

if [ ! -d "$MODEL_REPO" ]; then
    echo "错误: 模型仓库不存在: $MODEL_REPO"
    echo "请先运行: python scripts/export_whisper_to_onnx.py"
    exit 1
fi

echo "============================================================"
echo "  Triton Inference Server — Whisper"
echo "============================================================"
echo "  Docker Image : $TRITON_IMAGE"
echo "  Container    : $CONTAINER_NAME"
echo "  Model Repo   : $MODEL_REPO"
echo "  GPU          : $GPU_ID"
echo "  gRPC Port    : $GRPC_PORT"
echo "  HTTP Port    : $HTTP_PORT"
echo "  Metrics Port : $METRICS_PORT"
echo "  Log Level    : $LOG_LEVEL"
echo "============================================================"
echo ""

# ─── 检查 Docker ──────────────────────────────────────────
if ! command -v docker &> /dev/null; then
    echo "错误: Docker 未安装或不在 PATH 中"
    exit 1
fi

# ─── 检查 NVIDIA Container Toolkit ────────────────────────
if ! docker run --rm --gpus "$GPU_ID" "$TRITON_IMAGE" nvidia-smi &> /dev/null; then
    echo "错误: NVIDIA Container Toolkit 未配置或 GPU 不可用"
    echo "请确认:"
    echo "  1. nvidia-smi 在主机上可运行"
    echo "  2. nvidia-container-toolkit 已安装"
    echo "  3. Docker daemon 已重启"
    exit 1
fi

# ─── 停止已有容器 ──────────────────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[INFO] 停止已有容器: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# ─── 启动 Triton ───────────────────────────────────────────
echo "[INFO] 启动 Triton Inference Server..."
echo ""

docker run -d --rm \
    --name "$CONTAINER_NAME" \
    --gpus "\"device=${GPU_ID}\"" \
    -p "${GRPC_PORT}:8001" \
    -p "${HTTP_PORT}:8000" \
    -p "${METRICS_PORT}:8002" \
    -v "${MODEL_REPO}:/models:ro" \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    "$TRITON_IMAGE" \
    tritonserver \
        --model-repository=/models \
        --log-verbose="$LOG_LEVEL" \
        --strict-model-config=false \
        --grpc-infer-allocation-pool-size=64 \
        --response-cache-byte-size=1048576

# ─── 等待 Triton 就绪 ──────────────────────────────────────
echo ""
echo "[INFO] 等待 Triton 就绪 (最多 60s)..."

for i in $(seq 1 60); do
    if curl -s "http://localhost:${HTTP_PORT}/v2/health/ready" > /dev/null 2>&1; then
        echo ""
        echo "[OK] Triton 已就绪！"
        echo ""
        echo "  HTTP API : http://localhost:${HTTP_PORT}"
        echo "  gRPC API : localhost:${GRPC_PORT}"
        echo "  Metrics  : http://localhost:${METRICS_PORT}/metrics"
        echo ""
        echo "  检查模型状态:"
        curl -s "http://localhost:${HTTP_PORT}/v2/models/whisper_encoder/ready" && echo "  → encoder ready"
        curl -s "http://localhost:${HTTP_PORT}/v2/models/whisper_decoder/ready" && echo "  → decoder ready"
        echo ""
        echo "  查看日志: docker logs -f $CONTAINER_NAME"
        echo "  停止服务: docker stop $CONTAINER_NAME"
        exit 0
    fi
    sleep 1
    echo -n "."
done

echo ""
echo "[ERROR] Triton 启动超时。查看日志:"
echo "  docker logs $CONTAINER_NAME"
exit 1
