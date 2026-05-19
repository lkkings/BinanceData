"""多 symbol 历史回填编排"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone

from ..domain import UnifiedMarketData
from ..infrastructure.binance import HistoryDownloader
from ..infrastructure.storage import RedisStore, SQLiteStore
from ..logging import get_logger

logger = get_logger(__name__)


async def bootstrap_symbol(
    symbol: str,
    sqlite_store: SQLiteStore,
    redis_store: RedisStore,
    backfill_days: int,
    redis_retention_days: int,
) -> set[date]:
    """对单个 symbol 回填 T-1..T-N 天，返回未能下载的日期集合。"""
    today = datetime.now(tz=timezone.utc).date()
    target_dates = [today - timedelta(days=i) for i in range(1, backfill_days + 1)]

    downloader = HistoryDownloader(symbol)
    missing: set[date] = set()

    for d in target_dates:
        if await sqlite_store.has_full_day(symbol, d):
            logger.info("bootstrap.skip_existing", symbol=symbol, date=d.isoformat())
            if (today - d).days < redis_retention_days:
                start_ts = int(datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc).timestamp())
                end_ts = start_ts + 86400 - 1
                cached = await sqlite_store.range(symbol.upper(), start_ts, end_ts)
                if cached:
                    await redis_store.bulk_load(cached)
            continue

        date_str = d.strftime("%Y-%m-%d")
        try:
            records: list[UnifiedMarketData] | None = await asyncio.to_thread(
                downloader.collect_day_minutes, date_str
            )
        except Exception as e:
            logger.error("bootstrap.download_error", symbol=symbol, date=date_str, error=str(e))
            missing.add(d)
            continue

        if records is None:
            logger.warning("bootstrap.not_published", symbol=symbol, date=date_str)
            missing.add(d)
            continue
        if not records:
            continue

        await sqlite_store.upsert_many(records)
        if (today - d).days < redis_retention_days:
            await redis_store.bulk_load(records)
        logger.info(
            "bootstrap.day_loaded", symbol=symbol, date=date_str, count=len(records)
        )

    return missing


async def bootstrap_all_symbols(
    symbols: list[str],
    sqlite_store: SQLiteStore,
    redis_store: RedisStore,
    backfill_days: int,
    redis_retention_days: int,
) -> dict[str, set[date]]:
    """并发回填所有 symbol，返回 {symbol_upper: missing_dates}。"""
    results = await asyncio.gather(
        *(
            bootstrap_symbol(s, sqlite_store, redis_store, backfill_days, redis_retention_days)
            for s in symbols
        ),
        return_exceptions=True,
    )
    out: dict[str, set[date]] = {}
    for symbol, res in zip(symbols, results):
        if isinstance(res, Exception):
            logger.error("bootstrap.symbol_failed", symbol=symbol, error=str(res))
            out[symbol.upper()] = set()
        else:
            out[symbol.upper()] = res
    return out
