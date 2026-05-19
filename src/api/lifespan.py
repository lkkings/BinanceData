"""FastAPI 应用生命周期 — 启动顺序：

1. 配置日志。
2. 连接 SQLite + Redis。
3. 初始化 PubSub。
4. 启动 Pipeline（聚合器 + Binance WebSocket）。
5. 历史回填（异步并发，多 symbol）。
6. 注册 RetryScheduler。

关闭按反向顺序释放资源。
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, time, timezone
from typing import AsyncIterator

from fastapi import FastAPI

from ..config import Settings, get_settings
from ..core.bootstrap import bootstrap_all_symbols
from ..core.pipeline import MarketDataPipeline
from ..core.pubsub import PubSub
from ..infrastructure.scheduler import DailyRefreshScheduler, HistoryRetryScheduler, IntegrityChecker, OnDemandFetcher
from ..infrastructure.storage import RedisStore, SQLiteStore
from ..logging import configure_logging, get_logger
from .deps import Container

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)

    logger.info("app.boot", symbols=settings.symbol_list)

    sqlite_store = await SQLiteStore.connect(settings.sqlite_path)
    redis_store = await RedisStore.connect(
        settings.redis_url, retention_days=settings.redis_retention_days
    )
    pubsub = PubSub(queue_maxsize=settings.ws_client_queue_maxsize)

    pipeline = MarketDataPipeline(
        symbols=settings.symbol_list,
        streams=settings.streams,
        is_futures=settings.is_futures,
        sqlite_store=sqlite_store,
        redis_store=redis_store,
        pubsub=pubsub,
    )

    container = Container(
        settings=settings,
        sqlite_store=sqlite_store,
        redis_store=redis_store,
        pubsub=pubsub,
        pipeline=pipeline,
        on_demand_fetcher=OnDemandFetcher(
            sqlite_store=sqlite_store,
            redis_store=redis_store,
            retention_days=settings.redis_retention_days,
        ),
        retry_schedulers=[],
    )
    app.state.container = container

    # 1) 历史回填
    logger.info("bootstrap.start", days=settings.history_backfill_days)
    missing_by_symbol = await bootstrap_all_symbols(
        settings.symbol_list,
        sqlite_store,
        redis_store,
        settings.history_backfill_days,
        settings.redis_retention_days,
    )
    logger.info(
        "bootstrap.done",
        missing={k: [d.isoformat() for d in sorted(v)] for k, v in missing_by_symbol.items()},
    )

    # 2) 数据完整性检查（修复每天不足 1440 条的日期）
    logger.info("integrity.check.start")
    integrity_checker = IntegrityChecker(sqlite_store)
    repaired = await integrity_checker.check_and_repair(
        settings.symbol_list,
        retention_days=settings.history_backfill_days,
    )
    if repaired:
        logger.info(
            "integrity.repaired",
            repaired={k: [d.isoformat() for d in v] for k, v in repaired.items()},
        )
        # 修复的日期重新加载到 Redis（仅 retention 窗口内）
        today = datetime.now(tz=timezone.utc).date()
        for symbol, dates in repaired.items():
            for d in dates:
                if (today - d).days >= settings.redis_retention_days:
                    continue
                start_ts = int(datetime.combine(d, time.min, tzinfo=timezone.utc).timestamp())
                end_ts = start_ts + 86400 - 1
                records = await sqlite_store.range(symbol, start_ts, end_ts)
                if records:
                    await redis_store.bulk_load(records)
    else:
        logger.info("integrity.check.done", status="all_complete")

    # 3) 重试任务（用于 bootstrap 失败的日期）
    for symbol, missing in missing_by_symbol.items():
        if not missing:
            continue
        sched = HistoryRetryScheduler(
            symbol=symbol,
            missing_dates=missing,
            sqlite_store=sqlite_store,
            redis_store=redis_store,
            cron_hour=settings.history_retry_cron_hour,
            retention_days=settings.redis_retention_days,
        )
        sched.start()
        container.retry_schedulers.append(sched)

    # 4) 每日历史刷新（永久 cron，覆盖前一天实时数据）
    daily_refresh = DailyRefreshScheduler(
        symbols=settings.symbol_list,
        sqlite_store=sqlite_store,
        redis_store=redis_store,
        cron_hour=settings.history_retry_cron_hour,
        cron_minute=30,
        retention_days=settings.redis_retention_days,
    )
    daily_refresh.start()
    container.daily_refresh = daily_refresh

    # 4) 启动数据管道
    await pipeline.start()
    logger.info("app.ready", api_host=settings.api_host, api_port=settings.api_port)

    try:
        yield
    finally:
        logger.info("app.shutdown")
        if container.daily_refresh:
            container.daily_refresh.stop()
        for s in container.retry_schedulers:
            s.stop()
        await pipeline.stop()
        await pubsub.close_all()
        await redis_store.close()
        await sqlite_store.close()
        logger.info("app.stopped")
