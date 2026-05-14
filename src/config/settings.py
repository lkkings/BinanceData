"""应用配置"""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    # Binance API 配置（可选，用于私有数据）
    binance_api_key: str | None = None
    binance_api_secret: str | None = None

    # 交易对和数据流配置
    symbols: list[str] = ["btcusdt"]
    streams: list[str] = ["depth@100ms", "trade"]

    # 数据存储配置
    data_dir: Path = Path("./data")
    raw_data_dir: Path = Path("./data/raw")
    aggregated_data_dir: Path = Path("./data/aggregated")

    # 历史数据配置
    history_base_url: str = "https://data.binance.vision/data/futures/um/daily/trades"
    is_futures: bool = True

    # 日志配置
    log_level: str = "INFO"

    # WebSocket 配置
    ws_ping_interval: int = 20  # 心跳间隔（秒）
    ws_ping_timeout: int = 10   # 心跳超时（秒）
    reconnect_delay: int = 5    # 重连延迟（秒）

    # 聚合配置
    aggregation_interval: int = 1  # 聚合间隔（秒）

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保数据目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.aggregated_data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()
