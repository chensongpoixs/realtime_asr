#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# 启动 Triton Inference Server (Whisper ONNX 模型)
# ═══════════════════════════════════════════════════════════════
#
# 前置条件:
#   1. ONNX 模型已导出: python scripts/export_whisper_to_onnx.py
#   2. triton_model_repo/ 目录存在且包含 encoder/decoder config
#
# 两种运行模式:
#
#   Docker 模式 (默认):
#     需要: Docker + NVIDIA Container Toolkit
#     bash scripts/start_triton_server.sh
#     bash scripts/start_triton_server.sh --mode docker
#
#   Native 模式 (直接在 Linux 宿主机运行):
#     需要: tritonserver 已安装并位于 PATH
#     bash scripts/start_triton_server.sh --mode native
#     bash scripts/start_triton_server.sh --mode native --gpu 0 --port 8001
#
# 常用参数:
#   --mode docker|native   运行模式 (默认: docker)
#   --gpu 0                指定 GPU ID (默认: all)
#   --port 8001            gRPC 端口 (默认: 8001, HTTP=port-1, Metrics=port+1)
#   --log-level 0|1|2      日志级别 (0=MIN, 1=INFO, 2=VERBOSE; 默认: 1)
#
# 验证:
#   curl http://localhost:8000/v2/health/ready
#   curl http://localhost:8000/v2/models/whisper_encoder/ready
# ═══════════════════════════════════════════════════════════════

set -e

# ─── 默认配置 ──────────────────────────────────────────────
RUN_MODE="docker"                     # docker | native
TRITON_IMAGE="nvcr.io/nvidia/tritonserver:24.01-py3"
CONTAINER_NAME="triton_whisper"
GPU_ID="all"
GRPC_PORT="8001"
HTTP_PORT="8000"
METRICS_PORT="8002"
LOG_LEVEL="1"                         # 0=MIN, 1=INFO, 2=VERBOSE
TRITON_LOG_FILE=""                    # native 模式的日志文件路径

# ─── 解析参数 ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            RUN_MODE="$2"
            if [[ "$RUN_MODE" != "docker" && "$RUN_MODE" != "native" ]]; then
                echo "错误: --mode 必须是 docker 或 native"
                exit 1
            fi
            shift 2
            ;;
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
        --log-file)
            TRITON_LOG_FILE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "  运行模式:"
            echo "    --mode docker|native    运行模式 (默认: docker)"
            echo ""
            echo "  通用参数:"
            echo "    --gpu GPU_ID            GPU ID (默认: all)"
            echo "    --port GRPC_PORT        gRPC 端口 (默认: 8001)"
            echo "    --log-level 0|1|2       日志级别 (默认: 1)"
            echo ""
            echo "  Docker 模式:"
            echo "    --image IMAGE           Triton Docker 镜像"
            echo ""
            echo "  Native 模式:"
            echo "    --log-file PATH         Triton 日志输出文件 (默认: ./triton.log)"
            echo ""
            echo "  示例:"
            echo "    $0                              # Docker 模式, 使用所有 GPU"
            echo "    $0 --gpu 0 --port 9001          # Docker 模式, 仅 GPU 0, 端口 9001"
            echo "    $0 --mode native                # Native 模式, 前台运行"
            echo "    $0 --mode native --gpu 0        # Native 模式, 仅 GPU 0"
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

# ─── 通用 Triton 参数 ─────────────────────────────────────
TRITON_ARGS=(
    --model-repository="$MODEL_REPO"
    --log-verbose="$LOG_LEVEL"
    --strict-model-config=false
    --grpc-infer-allocation-pool-size=64
    --response-cache-byte-size=1048576
)







echo "============================================================"
echo "  Triton Inference Server — Whisper"
echo "============================================================"
echo "  运行模式     : $RUN_MODE"
echo "  模型仓库     : $MODEL_REPO"
echo "  GPU          : $GPU_ID"
echo "  gRPC 端口    : $GRPC_PORT"
echo "  HTTP 端口    : $HTTP_PORT"
echo "  Metrics 端口 : $METRICS_PORT"
echo "  日志级别     : $LOG_LEVEL"
echo "============================================================"
echo ""

# ═══════════════════════════════════════════════════════════════
# Native 模式
# ═══════════════════════════════════════════════════════════════

run_native() {
    echo "[INFO] Native 模式启动..."

    # ─── 检查 tritonserver ──────────────────────────────────
    if ! command -v tritonserver &> /dev/null; then
        echo "错误: tritonserver 未安装或不在 PATH 中"
        echo ""
        echo "安装方法:"
        echo "  1. pip install tritonclient[all]                   # Python 客户端"
        echo "  2. 下载 Triton Server:"
        echo "     https://github.com/triton-inference-server/server/releases"
        echo "     或通过 NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver"
        echo ""
        echo "  3. 从 Docker 镜像提取二进制 (不推荐但可行):"
        echo "     docker pull nvcr.io/nvidia/tritonserver:24.01-py3"
        echo "     docker run --rm -v /tmp:/mnt nvcr.io/nvidia/tritonserver:24.01-py3 \\"
        echo "       cp /opt/tritonserver/bin/tritonserver /mnt/"
        exit 1
    fi

    # ─── 检查 GPU ────────────────────────────────────────
    if [[ "$GPU_ID" != "all" ]]; then
        export CUDA_VISIBLE_DEVICES="$GPU_ID"
        echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    fi

    if ! nvidia-smi &> /dev/null; then
        echo "错误: nvidia-smi 不可用，请确认 GPU 驱动已安装"
        exit 1
    fi

    # ─── 检查端口冲突 ─────────────────────────────────────
    for p in "$GRPC_PORT" "$HTTP_PORT" "$METRICS_PORT"; do
        if ss -tlnp 2>/dev/null | grep -q ":$p " || netstat -tlnp 2>/dev/null | grep -q ":$p "; then
            echo "错误: 端口 $p 已被占用"
            echo "请使用 --port 指定其他端口"
            exit 1
        fi
    done

    # ─── 日志文件 ────────────────────────────────────────
    if [[ -z "$TRITON_LOG_FILE" ]]; then
        TRITON_LOG_FILE="$SCRIPT_DIR/../triton.log"
    fi

    # 创建日志目录
    mkdir -p "$(dirname "$TRITON_LOG_FILE")"

    echo "[INFO] 日志文件: $TRITON_LOG_FILE"
    echo "[INFO] 启动 tritonserver (后台运行)..."
    echo ""

    # ─── 启动 ─────────────────────────────────────────────
    nohup tritonserver \
        --http-port="$HTTP_PORT" \
        --grpc-port="$GRPC_PORT" \
        --metrics-port="$METRICS_PORT" \
        "${TRITON_ARGS[@]}" \
        > "$TRITON_LOG_FILE" 2>&1 &

    TRITON_PID=$!
    echo "[INFO] tritonserver PID: $TRITON_PID"
    echo "[CMD]  nohup tritonserver \
        --http-port=$HTTP_PORT \
        --grpc-port=$GRPC_PORT \
        --metrics-port=$METRICS_PORT \
        ${TRITON_ARGS[@]} \
        > $TRITON_LOG_FILE 2>&1 &"
    # 保存 PID 文件
    PID_FILE="$SCRIPT_DIR/../triton.pid"
    echo "$TRITON_PID" > "$PID_FILE"
    echo "[INFO] PID 文件: $PID_FILE"
}

# ═══════════════════════════════════════════════════════════════
# Docker 模式
# ═══════════════════════════════════════════════════════════════

run_docker() {
    echo "[INFO] Docker 模式启动..."

    # ─── 检查 Docker ──────────────────────────────────────
    if ! command -v docker &> /dev/null; then
        echo "错误: Docker 未安装或不在 PATH 中"
        echo ""
        echo "如果无法使用 Docker，请尝试 Native 模式:"
        echo "  bash $0 --mode native"
        exit 1
    fi

    # ─── 检查 NVIDIA Container Toolkit ────────────────────
    echo "[INFO] 检查 NVIDIA Container Toolkit..."
    if ! docker run --rm --gpus "\"device=${GPU_ID}\"" "$TRITON_IMAGE" nvidia-smi &> /dev/null; then
        echo "错误: NVIDIA Container Toolkit 未配置或 GPU 不可用"
        echo "请确认:"
        echo "  1. nvidia-smi 在主机上可运行"
        echo "  2. nvidia-container-toolkit 已安装"
        echo "  3. Docker daemon 已重启 (sudo systemctl restart docker)"
        echo ""
        echo "如果无法使用 Docker，请尝试 Native 模式:"
        echo "  bash $0 --mode native"
        exit 1
    fi
    echo "[INFO] Docker + GPU 环境检查通过"

    # ─── 停止已有容器 ──────────────────────────────────────
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "[INFO] 停止已有容器: $CONTAINER_NAME"
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
    fi

    # ─── 启动 Triton ───────────────────────────────────────
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
            "${TRITON_ARGS[@]}"

    echo "docker run -d --rm \
        --name $CONTAINER_NAME \
        --gpus device=${GPU_ID} \
        -p ${GRPC_PORT}:8001 \
        -p ${HTTP_PORT}:8000 \
        -p ${METRICS_PORT}:8002 \
        -v ${MODEL_REPO}:/models:ro \
        --shm-size=1g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        $TRITON_IMAGE \
        tritonserver \
        ${TRITON_ARGS[@]}"

}

# ═══════════════════════════════════════════════════════════════
# 等待就绪
# ═══════════════════════════════════════════════════════════════

wait_ready() {
    echo ""
    echo "[INFO] 等待 Triton 就绪 (最多 60s)..."

    for i in $(seq 1 60); do
        if curl -s "http://localhost:${HTTP_PORT}/v2/health/ready" > /dev/null 2>&1; then
            echo ""
            echo "============================================================"
            echo "  ✅ Triton 已就绪！"
            echo "============================================================"
            echo ""
            echo "  HTTP API    : http://localhost:${HTTP_PORT}"
            echo "  gRPC API    : localhost:${GRPC_PORT}"
            echo "  Metrics     : http://localhost:${METRICS_PORT}/metrics"
            echo ""
            echo "  模型状态:"
            if curl -sf "http://localhost:${HTTP_PORT}/v2/models/whisper_encoder/ready" > /dev/null 2>&1; then
                echo "    ✅ whisper_encoder 就绪"
            else
                echo "    ⚠️  whisper_encoder 加载中..."
            fi
            if curl -sf "http://localhost:${HTTP_PORT}/v2/models/whisper_decoder/ready" > /dev/null 2>&1; then
                echo "    ✅ whisper_decoder 就绪"
            else
                echo "    ⚠️  whisper_decoder 加载中..."
            fi
            echo ""

            case "$RUN_MODE" in
                docker)
                    echo "  查看日志: docker logs -f $CONTAINER_NAME"
                    echo "  停止服务: docker stop $CONTAINER_NAME"
                    ;;
                native)
                    echo "  查看日志: tail -f $TRITON_LOG_FILE"
                    echo "  停止服务: kill \$(cat $PID_FILE)"
                    echo "         或: pkill -f tritonserver"
                    ;;
            esac
            echo ""
            return 0
        fi
        sleep 1
        echo -n "."
    done

    echo ""
    echo "[ERROR] Triton 启动超时。"
    case "$RUN_MODE" in
        docker)
            echo "  查看日志: docker logs $CONTAINER_NAME"
            ;;
        native)
            echo "  查看日志: tail -50 $TRITON_LOG_FILE"
            echo "  PID 状态: $(kill -0 "$TRITON_PID" 2>&1 && echo '运行中' || echo '已退出')"
            ;;
    esac
    return 1
}

# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

case "$RUN_MODE" in
    docker)
        run_docker
        ;;
    native)
        run_native
        ;;
esac

wait_ready
