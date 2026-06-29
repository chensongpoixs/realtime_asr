"""
RealTime ASR 后端启动入口

用法:
    python run.py                     # 默认配置启动
    python run.py --port 8080         # 指定端口

替代原有的 python main.py 方式，提供更清晰的启动流程。
"""
import argparse
import os
import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# ⚠️ 关键步骤：必须在 import 任何 app 模块之前设置 HF_ENDPOINT
#
# huggingface_hub 在 import 时读取 HF_ENDPOINT 环境变量，且只读一次。
# 而 faster_whisper → huggingface_hub 会在 import Transcriber 时被触发。
# 所以这里先预读 YAML，设置好环境变量，然后再 import app 模块。
# ═══════════════════════════════════════════════════════════════
import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
    _preload_config = yaml.safe_load(_f)
_hf_endpoint = _preload_config.get("model", {}).get("hf_endpoint", "")
if _hf_endpoint:
    os.environ["HF_ENDPOINT"] = _hf_endpoint

# ─── 现在安全了：可以导入 app 模块 ────────────────────────────────
from app.main import app
from app.core.config import load_config
from app.utils.ssl_utils import get_local_ips, generate_self_signed_cert


def main():
    parser = argparse.ArgumentParser(description="RealTime ASR Backend")
    parser.add_argument("--port", type=int, default=None, help="服务端口（覆盖 config.yaml）")
    parser.add_argument("--host", type=str, default=None, help="绑定地址（覆盖 config.yaml）")
    args = parser.parse_args()

    cfg = load_config()
    server = cfg.get("server", {})
    host = args.host or server.get("host", "0.0.0.0")
    port = args.port or server.get("port", 9765)
    use_ssl = server.get("ssl_enabled", False)

    # ─── SSL 证书检查/生成 ──────────────────────────────────────
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

    ssl_certfile = str(cert_path.resolve()) if use_ssl else None
    ssl_keyfile = str(key_path.resolve()) if use_ssl else None

    # ─── 启动横幅 ──────────────────────────────────────────────
    proto = "https" if use_ssl else "http"
    ips = get_local_ips()
    lan_ips = [ip for ip in ips if ip not in ("127.0.0.1", "localhost") and not ip.startswith("127.")]

    print("\n" + "=" * 60)
    print("  RealTime ASR Backend")
    print(f"  {proto}://{host}:{port}")
    if use_ssl:
        print(f"  SSL 证书: {ssl_certfile}")
        for ip in lan_ips:
            print(f"  手机访问: https://{ip}:{port}")
    else:
        print("  协议: HTTP (SSL 未启用)")
    print("  日志级别: DEBUG")
    print("=" * 60 + "\n")

    # ─── 启动服务 ──────────────────────────────────────────────
    import uvicorn
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
    )


if __name__ == "__main__":
    main()
