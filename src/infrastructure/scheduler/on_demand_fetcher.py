"""按需历史数据拉取（客户端 backfill 时触发）

当客户端请求的时间范围在 SQLite/Redis 中无数据时，按天粒度从 Binance 历史下载。
内置去重锁：同一天同一 symbol 不会并发下载。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone

from ..binance.history_downloader import HistoryDownloader
from ..storage import RedisStore, SQLiteStore

logger = logging.getLogger(__name__)


class OnDemandFetcher:
    """按需拉取历史数据并写入存储。"""

    def __init__(
        self,
        sqlite_store: SQLiteStore,
        redis_store: RedisStore,
        retention_days: int = 7,
    ):
        self._sqlite = sqlite_store
        self._redis = redis_store
        self._retention_days = retention_days
        self._locks: dict[tuple[str, date], asyncio.Lock] = {}
        self._fetched: set[tuple[str, date]] = set()

    def _get_lock(self, symbol: str, day: date) -> asyncio.Lock:
        key = (symbol.upper(), day)
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    async def ensure_range(self, symbol: str, start_ts: int, end_ts: int) -> None:
        """确保指定时间范围的数据存在。缺失的天会从历史数据下载。

        只尝试 T-2 及更早的日期（T-1 可能尚未发布）。
        """
        today = datetime.now(tz=timezone.utc).date()
        start_day = datetime.fromtimestamp(start_ts, tz=timezone.utc).date()
        end_day = datetime.fromtimestamp(end_ts, tz=timezone.utc).date()

        # 只尝试 T-2 及更早（Binance 历史数据有 T+1/T+2 延迟）
        latest_available = today - timedelta(days=2)

        current = start_day
        tasks = []
        while current <= min(end_day, latest_available):
            key = (symbol.upper(), current)
            if key not in self._fetched:
                tasks.append(self._fetch_day_if_missing(symbol, current))
            current += timedelta(days=1)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_day_if_missing(self, symbol: str, day: date) -> None:
        key = (symbol.upper(), day)
        lock = self._get_lock(symbol, day)

        async with lock:
            if key in self._fetched:
                return

            if await self._sqlite.has_full_day(symbol, day):
                self._fetched.add(key)
                return

            date_str = day.strftime("%Y-%m-%d")
            logger.info(f"按需拉取历史数据: {symbol} {date_str}")

            downloader = HistoryDownloader(symbol)
            try:
                records = await asyncio.to_thread(downloader.collect_day_minutes, date_str)
            except Exception as e:
                logger.error(f"按需拉取失败 {symbol} {date_str}: {e}")
                return

            if records is None:
                logger.warning(f"按需拉取: {symbol} {date_str} 历史数据尚未发布")
                return
            if not records:
                return

            await self._sqlite.upsert_many(records)
            today = datetime.now(tz=timezone.utc).date()
            if (today - day).days < self._retention_days:
                await self._redis.bulk_load(records)

            self._fetched.add(key)
            logger.info(f"按需拉取完成: {symbol} {date_str} {len(records)} 条")
