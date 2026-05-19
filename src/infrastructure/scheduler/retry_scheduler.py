"""日定时重试调度器（APScheduler）"""
import asyncio
import logging
from datetime import date, datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from ..binance.history_downloader import HistoryDownloader
from ..storage import RedisStore, SQLiteStore

logger = logging.getLogger(__name__)


class HistoryRetryScheduler:
    """对未发布的历史日期，每天 02:00 UTC 重试，直到成功"""

    def __init__(
        self,
        symbol: str,
        missing_dates: set[date],
        sqlite_store: SQLiteStore,
        redis_store: RedisStore,
        cron_hour: int = 2,
        retention_days: int = 7,
    ):
        self.symbol = symbol.upper()
        self.missing_dates: set[date] = set(missing_dates)
        self.sqlite_store = sqlite_store
        self.redis_store = redis_store
        self.cron_hour = cron_hour
        self.retention_days = retention_days
        self._scheduler = AsyncIOScheduler(timezone="UTC")
        self._history = HistoryDownloader(symbol)
        self._job_id = f"history_retry_{self.symbol}"

    def start(self) -> None:
        if not self.missing_dates:
            logger.info("无缺失历史日期，跳过定时任务注册")
            return
        trigger = CronTrigger(hour=self.cron_hour, minute=0, timezone="UTC")
        self._scheduler.add_job(
            self._run, trigger, id=self._job_id, replace_existing=True
        )
        self._scheduler.start()
        logger.info(
            f"已注册历史回填重试任务: {sorted(self.missing_dates)} "
            f"(每日 {self.cron_hour:02d}:00 UTC)"
        )

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("历史回填重试任务已停止")

    async def _run(self) -> None:
        if not self.missing_dates:
            self._scheduler.remove_job(self._job_id)
            return

        logger.info(f"开始重试历史日期: {sorted(self.missing_dates)}")
        succeeded: set[date] = set()
        for d in sorted(self.missing_dates):
            try:
                records = await asyncio.to_thread(
                    self._history.collect_day_minutes, d.strftime("%Y-%m-%d")
                )
            except Exception as e:
                logger.error(f"{d} 重试异常: {e}", exc_info=True)
                continue
            if records is None:
                logger.info(f"{d} 历史数据仍未发布")
                continue
            if not records:
                continue
            await self.sqlite_store.upsert_many(records)
            if self._within_retention(d):
                await self.redis_store.bulk_load(records)
            logger.info(f"{d} 已回填 {len(records)} 条分钟数据")
            succeeded.add(d)

        self.missing_dates -= succeeded
        if not self.missing_dates:
            logger.info("所有缺失日期已回填，移除定时任务")
            try:
                self._scheduler.remove_job(self._job_id)
            except Exception:
                pass

    def _within_retention(self, d: date) -> bool:
        cutoff = datetime.now(tz=timezone.utc).date() - timedelta(days=self.retention_days)
        return d >= cutoff
