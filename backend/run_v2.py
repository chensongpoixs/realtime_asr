"""
RealTime ASR v2.0 后端启动入口 (Triton Inference Server)

用法:
    python run_v2.py                     # 默认配置启动 (port 9766)
    python run_v2.py --port 8080         # 指定端口
    python run_v2.py --config my_v2.yaml # 指定配置文件

与 v1 (run.py) 的关系:
- 加载 config_v2.yaml (不是 config.yaml)
- 导入 app_v2 (不是 app)
- 默认端口 9766 (不是 9765)
- 共享 SSL 证书和工具函数
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# 先配置一个临时的 root logger，用于启动阶段输出
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("realtime_asr.v2.boot")

# ═══════════════════════════════════════════════════════════════
# ⚠️ 关键步骤：必须在 import app 模块之前设置 HF_ENDPOINT
#
# Tokenizer 加载可能触发 huggingface_hub 导入，而它只读取一次 HF_ENDPOINT。
# 所以这里先预读 config_v2.yaml，设置好环境变量。
# ═══════════════════════════════════════════════════════════════
import yaml

BACKEND_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG = BACKEND_DIR / "config_v2.yaml"

# 解析 --config 参数 (手动预解析，因为此时还没导入应用)
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--config", type=str, default=None)
_pre_args, _ = _pre_parser.parse_known_args()

CONFIG_PATH = Path(_pre_args.config) if _pre_args.config else DEFAULT_CONFIG
if not CONFIG_PATH.is_absolute():
    CONFIG_PATH = BACKEND_DIR / CONFIG_PATH

if not CONFIG_PATH.exists():
    logger.error("[BOOT] 配置文件不存在: %s", CONFIG_PATH)
    sys.exit(1)

with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
    _preload_config = yaml.safe_load(_f)
_hf_endpoint = _preload_config.get("model", {}).get("hf_endpoint", "")
if _hf_endpoint:
    os.environ["HF_ENDPOINT"] = _hf_endpoint
    logger.info("[BOOT] HF_ENDPOINT 已预设: %s", _hf_endpoint)
else:
    logger.info("[BOOT] HF_ENDPOINT: (使用默认 huggingface.co)")

# ─── 现在安全了：可以导入 app 模块 ────────────────────────────────
from app.main_v2 import app_v2
from app.utils.ssl_utils import get_local_ips, generate_self_signed_cert


def main():
    parser = argparse.ArgumentParser(description="RealTime ASR v2.0 Backend (Triton)")
    parser.add_argument("--port", type=int, default=None,
                        help="服务端口（覆盖 config_v2.yaml）")
    parser.add_argument("--host", type=str, default=None,
                        help="绑定地址（覆盖 config_v2.yaml）")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径 (默认: config_v2.yaml)")
    args = parser.parse_args()

    # ─── 配置加载 ──────────────────────────────────────────────
    # 重新读取完整配置 (预读取只读了 model.hf_endpoint)
    logger.info("[BOOT] 加载配置: %s", CONFIG_PATH)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)

    server = full_config.get("server", {})
    triton_cfg = full_config.get("triton", {})
    trans_cfg = full_config.get("transcription", {})

    host = args.host or server.get("host", "0.0.0.0")
    port = args.port or server.get("port", 9766)
    use_ssl = server.get("ssl_enabled", False)

    # ─── SSL 证书检查/生成 (与 v1 共享) ───────────────────────
    ssl_cert = server.get("ssl_cert", "./fullchain.pem")
    ssl_key = server.get("ssl_key", "./privkey.pem")

    cert_path = Path(ssl_cert)
    key_path = Path(ssl_key)
    if not cert_path.is_absolute():
        cert_path = BACKEND_DIR / cert_path
    if not key_path.is_absolute():
        key_path = BACKEND_DIR / key_path

    # 证书总是生成（即使 SSL 未启用），方便用户下载安装后再切 HTTPS
    need_gen = False
    if not cert_path.exists() or not key_path.exists():
        logger.info("[BOOT] 证书文件不存在，准备自动生成...")
        need_gen = True
    else:
        try:
            first_line = cert_path.read_text(encoding="utf-8").strip().split("\n")[0]
            if "PRIVATE KEY" in first_line:
                logger.warning("[BOOT] %s 是私钥不是证书，重新生成...", ssl_cert)
                cert_path.unlink()
                key_path.unlink()
                need_gen = True
        except Exception as e:
            logger.warning("[BOOT] 证书文件读取异常: %s，重新生成...", e)
            need_gen = True

    if need_gen:
        logger.info("[BOOT] 生成自签名证书...")
        success = generate_self_signed_cert(str(cert_path), str(key_path))
        if not success:
            logger.error("[BOOT] 证书生成失败")
            sys.exit(1)

    ssl_certfile = str(cert_path.resolve()) if use_ssl else None
    ssl_keyfile = str(key_path.resolve()) if use_ssl else None

    # ─── 启动横幅 ──────────────────────────────────────────────
    proto = "https" if use_ssl else "http"
    ips = get_local_ips()
    lan_ips = [ip for ip in ips if ip not in ("127.0.0.1", "localhost") and not ip.startswith("127.")]

    print("\n" + "=" * 60)
    print("  RealTime ASR Backend v2.0 (Triton)")
    print(f"  {proto}://{host}:{port}")
    print(f"  WebSocket: {'wss' if use_ssl else 'ws'}://{host}:{port}/ws/transcribe")
    print()
    print(f"  Triton Server: {triton_cfg.get('url', 'localhost:8001')} (gRPC)")
    print(f"  模型: {triton_cfg.get('model_name', 'whisper_large_v3')}")
    print(f"  语言: {trans_cfg.get('language', 'auto')}")
    print(f"  Beam Size: {trans_cfg.get('beam_size', 5)}")
    print(f"  Buffer: {trans_cfg.get('buffer_threshold', 1.0)}s")
    print()

    if use_ssl:
        print(f"  SSL 证书: {ssl_certfile}")
        for ip in lan_ips:
            print(f"  手机访问: https://{ip}:{port}")
    else:
        print("  协议: HTTP (SSL 未启用)")

    print()
    print("  ★ 确保 Triton Inference Server 已启动")
    print("    docker run --gpus all -p 8001:8001 \\")
    print("      -v $(pwd)/triton_model_repo:/models \\")
    print("      nvcr.io/nvidia/tritonserver:24.01-py3 \\")
    print("      tritonserver --model-repository=/models")
    print("=" * 60 + "\n")

    logger.info("[BOOT] ========== v2.0 启动配置 ==========")
    logger.info("[BOOT] 服务: host=%s, port=%s, ssl=%s", host, port, use_ssl)
    logger.info("[BOOT] Triton: url=%s, model=%s",
                triton_cfg.get("url", "localhost:8001"),
                triton_cfg.get("model_name", "whisper_large_v3"))
    logger.info("[BOOT] 转录: language=%s, beam=%d, buffer=%.1fs",
                trans_cfg.get("language", "auto"),
                trans_cfg.get("beam_size", 5),
                trans_cfg.get("buffer_threshold", 1.0))
    if lan_ips:
        logger.info("[BOOT] 局域网 IP: %s", lan_ips)
    logger.info("[BOOT] 启动 uvicorn...")

    # ─── 启动服务 ──────────────────────────────────────────────
    import uvicorn
    uvicorn.run(
        app_v2,
        host=host,
        port=port,
        log_level="info",
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
    )


if __name__ == "__main__":
    main()
