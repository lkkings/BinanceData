"""进程内 Pub-Sub 总线

设计要点：
- 多客户端订阅，按 (symbol × channel) 分发。
- 每个订阅独立的 bounded asyncio.Queue；满时丢弃最旧元素，避免慢客户端拖累采集主循环。
- 订阅是 add/remove 安全的（订阅在 WS 端点 finally 中显式清理）。
- 仅做事件分发，不感知具体消息载荷类型——payload 由调用方负责序列化。
"""
from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import Iterable

from ..logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class Message:
    channel: str
    symbol: str
    payload: dict


@dataclass(slots=True, eq=False)
class Subscription:
    """单个客户端的订阅句柄。

    - symbols: 空集 = 任意 symbol（仅频道过滤）。
    - channels: 客户端关心的频道集合，例如 {"minute"}。
    - queue: 实时推送通道。
    - dropped: 因背压被丢弃的累计消息数（可观测）。
    """

    symbols: set[str] = field(default_factory=set)
    channels: set[str] = field(default_factory=set)
    queue: asyncio.Queue[Message] = field(default_factory=asyncio.Queue)
    dropped: int = 0
    closed: bool = False

    def matches(self, channel: str, symbol: str) -> bool:
        if self.closed:
            return False
        if channel not in self.channels:
            return False
        return not self.symbols or symbol.lower() in self.symbols


class PubSub:
    """全局事件总线 — 单例语义，绑定到 FastAPI app.state。"""

    def __init__(self, queue_maxsize: int = 1000):
        self._subs: set[Subscription] = set()
        self._queue_maxsize = queue_maxsize
        self._lock = asyncio.Lock()

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)

    def subscribe(
        self,
        channels: Iterable[str],
        symbols: Iterable[str] | None = None,
        queue_maxsize: int | None = None,
    ) -> Subscription:
        sub = Subscription(
            symbols={s.lower() for s in (symbols or ())},
            channels=set(channels),
            queue=asyncio.Queue(maxsize=queue_maxsize or self._queue_maxsize),
        )
        self._subs.add(sub)
        logger.debug("pubsub.subscribe", channels=list(sub.channels), symbols=list(sub.symbols))
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        sub.closed = True
        self._subs.discard(sub)
        logger.debug("pubsub.unsubscribe", dropped=sub.dropped)

    def update_subscription(
        self,
        sub: Subscription,
        *,
        add_symbols: Iterable[str] | None = None,
        remove_symbols: Iterable[str] | None = None,
        add_channels: Iterable[str] | None = None,
        remove_channels: Iterable[str] | None = None,
    ) -> None:
        if add_symbols:
            sub.symbols |= {s.lower() for s in add_symbols}
        if remove_symbols:
            sub.symbols -= {s.lower() for s in remove_symbols}
        if add_channels:
            sub.channels |= set(add_channels)
        if remove_channels:
            sub.channels -= set(remove_channels)

    async def publish(self, channel: str, symbol: str, payload: dict) -> None:
        if not self._subs:
            return
        msg = Message(channel=channel, symbol=symbol.lower(), payload=payload)
        for sub in list(self._subs):
            if not sub.matches(channel, symbol):
                continue
            try:
                sub.queue.put_nowait(msg)
            except asyncio.QueueFull:
                # 背压策略：丢最旧
                with contextlib.suppress(asyncio.QueueEmpty):
                    sub.queue.get_nowait()
                    sub.queue.task_done()
                with contextlib.suppress(asyncio.QueueFull):
                    sub.queue.put_nowait(msg)
                sub.dropped += 1

    async def close_all(self) -> None:
        async with self._lock:
            for sub in list(self._subs):
                sub.closed = True
            self._subs.clear()
