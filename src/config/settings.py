"""应用配置（统一入口）"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置（环境变量优先，其次 .env 文件，最后默认值）"""

    # === 数据采集 ===
    symbols: str = "btcusdt"  # 逗号分隔，例如 "btcusdt,ethusdt"
    streams: list[str] = ["depth@100ms", "trade"]
    is_futures: bool = True

    # === Binance 凭据（可选，预留）===
    binance_api_key: str | None = None
    binance_api_secret: str | None = None

    # === 存储 ===
    data_dir: Path = Path("./data")
    raw_data_dir: Path = Path("./data/raw")
    sqlite_path: Path = Path("./data/market.db")
    redis_url: str = "redis://localhost:6379/0"
    redis_retention_days: int = 7

    # === 历史 ===
    history_base_url: str = "https://data.binance.vision/data/futures/um/daily/trades"
    history_backfill_days: int = 7
    history_retry_cron_hour: int = 2  # UTC

    # === HTTP/WS 服务 ===
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""  # 空 = 不鉴权
    ws_max_history_minutes: int = 7 * 1440
    ws_client_queue_maxsize: int = 1000
    ws_default_history_minutes: int = 60

    # === Binance WebSocket 客户端 ===
    ws_ping_interval: int = 20
    ws_ping_timeout: int = 10
    reconnect_delay: int = 5

    # === 日志 ===
    log_level: str = "INFO"
    log_json: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def symbol_list(self) -> list[str]:
        return [s.strip().lower() for s in self.symbols.split(",") if s.strip()]

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.raw_data_dir, self.sqlite_path.parent):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
