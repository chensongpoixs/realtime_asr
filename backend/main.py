"""
向后兼容入口 — 请使用 python run.py 启动

保留此文件以兼容:
  - uvicorn main:app
  - python main.py (现在推荐使用 python run.py)
"""
import os
from pathlib import Path

# 预读配置设置 HF_ENDPOINT（必须在 import app 之前）
import yaml
_CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _preload = yaml.safe_load(_f)
_hf = _preload.get("model", {}).get("hf_endpoint", "")
if _hf:
    os.environ.setdefault("HF_ENDPOINT", _hf)

from app.main import app

if __name__ == "__main__":
    import uvicorn
    cfg = _preload
    server = cfg.get("server", {})
    uvicorn.run(
        "main:app",
        host=server.get("host", "0.0.0.0"),
        port=server.get("port", 9765),
        log_level="info",
        reload=False,
    )
