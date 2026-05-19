"""服务入口 — 仅启动 uvicorn"""
import uvicorn

from src.api.app import create_app
from src.config import get_settings
from src.logging import configure_logging

# 在 uvicorn 接管前先配一次日志，避免 reloader 导致两套配置交错
_settings = get_settings()
configure_logging(level=_settings.log_level, json_output=_settings.log_json)

app = create_app()


def main() -> None:
    uvicorn.run(
        "main:app",
        host=_settings.api_host,
        port=_settings.api_port,
        log_config=None,  # 沿用我们自己的 structlog 配置
        access_log=False,
    )


if __name__ == "__main__":
    main()
