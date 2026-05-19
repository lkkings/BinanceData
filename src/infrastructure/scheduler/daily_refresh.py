"""每日历史数据刷新调度器

每天 02:00 UTC 用 Binance 官方历史数据覆盖前一天的实时采集数据。
历史数据更权威（无断连、无幽灵事件），用 INSERT OR REPLACE 覆盖。
此 job 永久运行，不会自动移除。
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from ..binance.history_downloader import HistoryDownloader
from ..storage import RedisStore, SQLiteStore

logger = logging.getLogger(__name__)


class DailyRefreshScheduler:
    """每天用历史数据覆盖前一天的实时数据（永久 cron job）"""

    def __init__(
        self,
        symbols: list[str],
        sqlite_store: SQLiteStore,
        redis_store: RedisStore,
        cron_hour: int = 2,
        cron_minute: int = 30,
        retention_days: int = 7,
    ):
        self._symbols = [s.upper() for s in symbols]
        self._sqlite = sqlite_store
        self._redis = redis_store
        self._retention_days = retention_days
        self._scheduler = AsyncIOScheduler(timezone="UTC")
        self._cron_hour = cron_hour
        self._cron_minute = cron_minute

    def start(self) -> None:
        trigger = CronTrigger(
            hour=self._cron_hour, minute=self._cron_minute, timezone="UTC"
        )
        self._scheduler.add_job(
            self._refresh_yesterday,
            trigger,
            id="daily_refresh",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info(
            f"每日历史刷新已注册: symbols={self._symbols} "
            f"(每日 {self._cron_hour:02d}:{self._cron_minute:02d} UTC)"
        )

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    async def _refresh_yesterday(self) -> None:
        yesterday = (datetime.now(tz=timezone.utc) - timedelta(days=1)).date()
        date_str = yesterday.strftime("%Y-%m-%d")
        logger.info(f"开始每日历史刷新: {date_str} symbols={self._symbols}")

        for symbol in self._symbols:
            await self._refresh_day(symbol, date_str, yesterday)

    async def _refresh_day(self, symbol: str, date_str: str, day) -> None:
        downloader = HistoryDownloader(symbol)
        try:
            records = await asyncio.to_thread(downloader.collect_day_minutes, date_str)
        except Exception as e:
            logger.error(f"每日刷新失败 {symbol} {date_str}: {e}", exc_info=True)
            return

        if records is None:
            logger.warning(f"每日刷新: {symbol} {date_str} 历史数据尚未发布")
            return
        if not records:
            return

        count = await self._sqlite.upsert_many(records)
        today = datetime.now(tz=timezone.utc).date()
        if (today - day).days < self._retention_days:
            await self._redis.bulk_load(records)
        logger.info(f"每日刷新完成: {symbol} {date_str} 覆盖 {count} 条分钟数据")
