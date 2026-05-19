"""市场数据管道：WS → 聚合器 → (SQLite + Redis + PubSub)"""
from __future__ import annotations

import asyncio
from typing import Any

from ..domain import UnifiedMarketData
from ..infrastructure.binance import BinanceCollector
from ..infrastructure.storage import RedisStore, SQLiteStore
from ..infrastructure.storage.serialize import to_payload_dict
from ..logging import get_logger
from .aggregators import MinuteAggregator
from .pubsub import PubSub

logger = get_logger(__name__)


class MarketDataPipeline:
    """将上游 WebSocket 流串到聚合器和下游存储/广播。

    生命周期：
        await pipeline.start()
        await pipeline.run()       # 阻塞，直到被 stop()/取消
        await pipeline.stop()
    """

    def __init__(
        self,
        symbols: list[str],
        streams: list[str],
        is_futures: bool,
        sqlite_store: SQLiteStore,
        redis_store: RedisStore,
        pubsub: PubSub,
    ):
        self._symbols = [s.lower() for s in symbols]
        self._sqlite = sqlite_store
        self._redis = redis_store
        self._pubsub = pubsub
        self._aggregator = MinuteAggregator(on_aggregated_data=self._on_minute_closed)
        self._collector = BinanceCollector(
            symbols=self._symbols,
            streams=streams,
            is_futures=is_futures,
            on_message=self._on_ws_message,
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._collector_task: asyncio.Task[Any] | None = None
        self._running = False

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    async def start(self) -> None:
        if self._running:
            return
        self._loop = asyncio.get_running_loop()
        await self._aggregator.start()
        self._collector_task = asyncio.create_task(self._collector.start(), name="binance-ws")
        self._running = True
        logger.info("pipeline.started", symbols=self._symbols)

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        await self._collector.stop()
        if self._collector_task:
            self._collector_task.cancel()
            try:
                await self._collector_task
            except (asyncio.CancelledError, Exception):
                pass
        await self._aggregator.stop()
        logger.info("pipeline.stopped")

    def _on_ws_message(self, stream_type: str, data: dict) -> None:
        try:
            if stream_type.startswith("depth"):
                parsed = BinanceCollector.parse_depth_update(data)
                self._aggregator.add_orderbook_update(parsed["symbol"], parsed)
            elif stream_type == "trade":
                parsed = BinanceCollector.parse_trade(data)
                if parsed is None:
                    return
                self._aggregator.add_trade(parsed["symbol"], parsed)
            # kline 流不再消费：kline 字段从 trade 直接重建
        except Exception as e:
            logger.error("pipeline.parse_error", error=str(e), stream_type=stream_type)

    def _on_minute_closed(self, record: UnifiedMarketData) -> None:
        # Aggregator 在事件循环线程内回调，可直接 schedule
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._persist_and_broadcast(record), self._loop)

    async def _persist_and_broadcast(self, record: UnifiedMarketData) -> None:
        try:
            await self._sqlite.upsert(record)
        except Exception as e:
            logger.error("pipeline.sqlite_error", error=str(e), symbol=record.symbol)

        try:
            await self._redis.upsert(record)
        except Exception as e:
            logger.error("pipeline.redis_error", error=str(e), symbol=record.symbol)

        try:
            await self._pubsub.publish("minute", record.symbol, to_payload_dict(record))
        except Exception as e:
            logger.error("pipeline.publish_error", error=str(e), symbol=record.symbol)

        logger.info(
            "pipeline.minute",
            symbol=record.symbol,
            ts=record.timestamp.isoformat(),
            close=str(record.close),
            volume=str(record.volume),
        )
