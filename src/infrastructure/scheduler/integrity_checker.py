"""数据完整性检查器

启动时检查每天是否有完整的 1440 条分钟数据，缺失则从历史 API 补全。
"""
import asyncio
import logging
from datetime import date, datetime, timedelta, timezone

from ..binance.history_downloader import HistoryDownloader
from ..storage import SQLiteStore

logger = logging.getLogger(__name__)


class IntegrityChecker:
    """数据完整性检查器：确保每天有完整的 1440 条分钟数据"""

    def __init__(self, sqlite_store: SQLiteStore):
        self._sqlite = sqlite_store

    async def check_and_repair(
        self, symbols: list[str], retention_days: int
    ) -> dict[str, list[date]]:
        """检查并修复数据完整性

        Returns:
            {symbol: [repaired_dates]} — 每个 symbol 修复的日期列表
        """
        today = datetime.now(tz=timezone.utc).date()
        start_date = today - timedelta(days=retention_days)

        repaired: dict[str, list[date]] = {}

        for symbol in symbols:
            symbol_upper = symbol.upper()
            logger.info(f"检查 {symbol_upper} 数据完整性 ({start_date} ~ {today - timedelta(days=1)})")

            incomplete_dates = await self._find_incomplete_dates(
                symbol_upper, start_date, today
            )

            if not incomplete_dates:
                logger.info(f"{symbol_upper} 数据完整")
                continue

            logger.warning(
                f"{symbol_upper} 发现 {len(incomplete_dates)} 天数据不完整: "
                f"{[d.isoformat() for d in sorted(incomplete_dates)[:5]]}{'...' if len(incomplete_dates) > 5 else ''}"
            )

            repaired_dates = await self._repair_dates(symbol_upper, incomplete_dates)
            if repaired_dates:
                repaired[symbol_upper] = repaired_dates

        return repaired

    async def _find_incomplete_dates(
        self, symbol: str, start_date: date, end_date: date
    ) -> list[date]:
        """查找数据不完整的日期（不足 1440 条分钟数据）"""
        incomplete: list[date] = []
        current = start_date

        while current < end_date:
            count = await self._sqlite.count_day(symbol, current)
            if count < 1440:
                incomplete.append(current)
            current += timedelta(days=1)

        return incomplete

    async def _repair_dates(
        self, symbol: str, dates: list[date]
    ) -> list[date]:
        """修复不完整的日期：从历史 API 下载并覆盖"""
        downloader = HistoryDownloader(symbol)
        repaired: list[date] = []

        for d in sorted(dates):
            date_str = d.strftime("%Y-%m-%d")
            try:
                records = await asyncio.to_thread(
                    downloader.collect_day_minutes, date_str
                )
            except Exception as e:
                logger.error(f"修复失败 {symbol} {date_str}: {e}")
                continue

            if records is None:
                logger.warning(f"修复跳过 {symbol} {date_str}: 历史数据尚未发布")
                continue

            if not records:
                logger.warning(f"修复跳过 {symbol} {date_str}: 聚合后为空")
                continue

            count = await self._sqlite.upsert_many(records)
            logger.info(f"修复完成 {symbol} {date_str}: 覆盖 {count} 条")
            repaired.append(d)

        return repaired
