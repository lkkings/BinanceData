"""领域协议（可被 core/api 层依赖，可被测试用内存实现替换）"""
from typing import Protocol, runtime_checkable

from .aggregated import UnifiedMarketData


@runtime_checkable
class IStorage(Protocol):
    """统一的市场数据存储接口"""

    async def upsert(self, record: UnifiedMarketData) -> None: ...

    async def upsert_many(self, records: list[UnifiedMarketData]) -> int: ...

    async def range(
        self, symbol: str, start_ts: int, end_ts: int
    ) -> list[UnifiedMarketData]: ...

    async def close(self) -> None: ...


@runtime_checkable
class IBroadcaster(Protocol):
    """进程内 pub-sub 接口（核心层只依赖此抽象）"""

    async def publish(self, channel: str, symbol: str, payload: dict) -> None: ...
