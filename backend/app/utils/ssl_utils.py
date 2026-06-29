"""
SSL 工具 — 自签名证书生成 + 本机 IP 检测

用于开发环境在局域网内通过 HTTPS/WSS 访问（iOS 要求 getUserMedia 必须 HTTPS）。
"""
import logging
import socket
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger("realtime_asr")


def get_local_ips() -> list[str]:
    """获取本机所有局域网 IP 和主机名"""
    ips = ["127.0.0.1", "localhost"]
    try:
        hostname = socket.gethostname()
        ips.append(hostname)
        ips.append(f"{hostname}.local")
    except Exception:
        pass
    try:
        # 获取默认路由对应的 IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return list(set(ips))


def generate_self_signed_cert(cert_path: str, key_path: str) -> bool:
    """生成带 SAN 的自签名证书，解决 iOS WSS 证书主机名不匹配问题"""
    ips = get_local_ips()

    # 构建 SAN 扩展：DNS + IP
    san_entries = []
    for ip in ips:
        try:
            socket.inet_aton(ip)
            san_entries.append(f"IP:{ip}")
        except (socket.error, OSError):
            san_entries.append(f"DNS:{ip}")

    san_ext = ",".join(san_entries)
    logger.info("[SSL] 生成自签名证书, SAN: %s", san_ext)

    # 创建临时配置文件（openssl 需要配置文件来指定 SAN）
    config = f"""
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = RealTime ASR

[v3_req]
subjectAltName = {san_ext}
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
"""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cnf", delete=False) as f:
            f.write(config)
            config_file = f.name

        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", key_path,
                "-out", cert_path,
                "-days", "3650",
                "-nodes",
                "-config", config_file,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        Path(config_file).unlink()
        logger.info("[SSL] 证书已生成: %s", cert_path)
        logger.info("[SSL] 私钥已生成: %s", key_path)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("[SSL] 证书生成失败: %s", e.stderr)
        return False
    except FileNotFoundError:
        logger.error("[SSL] 未安装 openssl，无法生成证书。请安装: apt install openssl")
        return False
