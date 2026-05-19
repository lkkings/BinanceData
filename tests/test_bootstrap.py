"""Bootstrap 编排测试 — mock HistoryDownloader 以避开外网"""
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.bootstrap import bootstrap_all_symbols, bootstrap_symbol
from src.domain import UnifiedMarketData
from src.infrastructure.storage import RedisStore, SQLiteStore

pytestmark = pytest.mark.asyncio


def _make_minute_records(symbol: str, day: date, minutes: int = 1440) -> list[UnifiedMarketData]:
    base = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
    return [
        UnifiedMarketData(
            timestamp=base + timedelta(minutes=i),
            symbol=symbol.upper(),
            open=Decimal("100"),
            high=Decimal("100"),
            low=Decimal("100"),
            close=Decimal("100"),
            volume=Decimal("1"),
        )
        for i in range(minutes)
    ]


@pytest.fixture
async def stores(tmp_path: Path):
    sqlite = await SQLiteStore.connect(tmp_path / "test.db")
    redis = await RedisStore.connect("fake://", retention_days=7)
    try:
        yield sqlite, redis
    finally:
        await redis.close()
        await sqlite.close()


async def test_bootstrap_all_days_succeed(stores):
    sqlite, redis = stores
    today = datetime.now(tz=timezone.utc).date()

    def fake_collect(self, date_str: str):
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return _make_minute_records(self.symbol, d, minutes=1440)

    with patch(
        "src.infrastructure.binance.HistoryDownloader.collect_day_minutes",
        new=fake_collect,
    ):
        missing = await bootstrap_symbol(
            "btcusdt",
            sqlite,
            redis,
            backfill_days=3,
            redis_retention_days=7,
        )

    assert missing == set()
    # SQLite 应有 3 天 × 1440 条
    rng = await sqlite.range("BTCUSDT", 0, 10**12)
    assert len(rng) == 3 * 1440


async def test_bootstrap_collects_missing_dates(stores):
    sqlite, redis = stores
    today = datetime.now(tz=timezone.utc).date()
    yesterday = today - timedelta(days=1)
    two_days_ago = today - timedelta(days=2)

    def fake_collect(self, date_str: str):
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        # 模拟最近两天历史数据未发布
        if d in (yesterday, two_days_ago):
            return None
        return _make_minute_records(self.symbol, d, minutes=1440)

    with patch(
        "src.infrastructure.binance.HistoryDownloader.collect_day_minutes",
        new=fake_collect,
    ):
        missing = await bootstrap_symbol(
            "btcusdt",
            sqlite,
            redis,
            backfill_days=4,
            redis_retention_days=7,
        )

    assert missing == {yesterday, two_days_ago}
    # 剩下两天应已落库 (T-3, T-4)
    rng = await sqlite.range("BTCUSDT", 0, 10**12)
    assert len(rng) == 2 * 1440


async def test_bootstrap_skips_when_sqlite_has_full_day(stores):
    sqlite, redis = stores
    today = datetime.now(tz=timezone.utc).date()
    target_day = today - timedelta(days=2)

    # 预先把 T-2 写满
    await sqlite.upsert_many(_make_minute_records("BTCUSDT", target_day, minutes=1440))

    call_count = {"n": 0}

    def fake_collect(self, date_str: str):
        call_count["n"] += 1
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return _make_minute_records(self.symbol, d, minutes=1440)

    with patch(
        "src.infrastructure.binance.HistoryDownloader.collect_day_minutes",
        new=fake_collect,
    ):
        await bootstrap_symbol("btcusdt", sqlite, redis, backfill_days=3, redis_retention_days=7)

    # 应该只下载 T-1 和 T-3，不重复下载已存在的 T-2
    assert call_count["n"] == 2


async def test_bootstrap_all_symbols_in_parallel(stores):
    sqlite, redis = stores

    def fake_collect(self, date_str: str):
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return _make_minute_records(self.symbol, d, minutes=1440)

    with patch(
        "src.infrastructure.binance.HistoryDownloader.collect_day_minutes",
        new=fake_collect,
    ):
        missing_by = await bootstrap_all_symbols(
            ["btcusdt", "ethusdt"],
            sqlite,
            redis,
            backfill_days=2,
            redis_retention_days=7,
        )

    assert set(missing_by.keys()) == {"BTCUSDT", "ETHUSDT"}
    assert all(v == set() for v in missing_by.values())

    btc_count = len(await sqlite.range("BTCUSDT", 0, 10**12))
    eth_count = len(await sqlite.range("ETHUSDT", 0, 10**12))
    assert btc_count == 2 * 1440
    assert eth_count == 2 * 1440
