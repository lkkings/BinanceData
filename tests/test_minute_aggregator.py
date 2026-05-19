"""MinuteAggregator 单元测试"""
import asyncio
from decimal import Decimal

import pytest

from src.core.aggregators import MinuteAggregator

pytestmark = pytest.mark.asyncio


def make_trade(symbol: str, ms: int, price: str, qty: str = "1", is_buyer_maker: bool = False):
    return {
        "event_time": ms,
        "symbol": symbol,
        "trade_id": ms,
        "price": Decimal(price),
        "quantity": Decimal(qty),
        "trade_time": ms,
        "is_buyer_maker": is_buyer_maker,
    }


async def test_aggregate_single_minute_ohlcv():
    received = []
    agg = MinuteAggregator(on_aggregated_data=received.append, watermark_delay_seconds=1)
    await agg.start()
    try:
        base = 1779107400000  # 2026-05-18 12:30:00 UTC
        for i in range(5):
            agg.add_trade("BTCUSDT", make_trade("BTCUSDT", base + i * 100, str(100 + i)))

        # 让聚合器先初始化 last_flushed_minute
        await asyncio.sleep(1.2)

        # 推进 watermark 到下一分钟
        agg.add_trade("BTCUSDT", make_trade("BTCUSDT", base + 60000, "200"))
        await asyncio.sleep(1.5)
    finally:
        await agg.stop()

    minute_30 = [r for r in received if r.timestamp.minute == 30]
    assert len(minute_30) == 1
    r = minute_30[0]
    assert r.symbol == "BTCUSDT"
    assert r.open == Decimal("100")
    assert r.close == Decimal("104")
    assert r.high == Decimal("104")
    assert r.low == Decimal("100")
    assert r.volume == Decimal("5")
    assert r.trade_count == 5


async def test_multi_symbol_isolation():
    received = []
    agg = MinuteAggregator(on_aggregated_data=received.append, watermark_delay_seconds=1)
    await agg.start()
    try:
        base = 1779107400000
        for i in range(3):
            agg.add_trade("BTCUSDT", make_trade("BTCUSDT", base + i * 100, "100"))
            agg.add_trade("ETHUSDT", make_trade("ETHUSDT", base + i * 100, "10"))

        await asyncio.sleep(1.2)
        agg.add_trade("BTCUSDT", make_trade("BTCUSDT", base + 60000, "200"))
        await asyncio.sleep(1.5)
    finally:
        await agg.stop()

    by_symbol = {r.symbol: r for r in received if r.timestamp.minute == 30}
    assert "BTCUSDT" in by_symbol and "ETHUSDT" in by_symbol
    assert by_symbol["BTCUSDT"].close == Decimal("100")
    assert by_symbol["ETHUSDT"].close == Decimal("10")
