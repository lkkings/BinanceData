"""依赖注入容器 — 在 lifespan 中实例化，挂在 app.state.container"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..core.pipeline import MarketDataPipeline
from ..core.pubsub import PubSub
from ..infrastructure.scheduler import DailyRefreshScheduler, HistoryRetryScheduler, OnDemandFetcher
from ..infrastructure.storage import RedisStore, SQLiteStore


@dataclass
class Container:
    settings: Settings
    sqlite_store: SQLiteStore
    redis_store: RedisStore
    pubsub: PubSub
    pipeline: MarketDataPipeline
    on_demand_fetcher: OnDemandFetcher
    daily_refresh: DailyRefreshScheduler | None = None
    retry_schedulers: list[HistoryRetryScheduler] = field(default_factory=list)


def get_container(request) -> Container:
    """FastAPI 依赖：从 app.state 取容器。"""
    return request.app.state.container
