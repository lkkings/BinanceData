"""Redis 热缓存存储 (Hash + Sorted Set 双结构)"""
import logging
from typing import Iterable

import redis.asyncio as aioredis

from ...domain import UnifiedMarketData
from .serialize import epoch_seconds, from_json, to_json

logger = logging.getLogger(__name__)


def _zindex_key(symbol: str) -> str:
    return f"md:{symbol.upper()}:zindex"


def _data_key(symbol: str) -> str:
    return f"md:{symbol.upper()}:data"


class RedisStore:
    """基于 Hash(数据) + Sorted Set(时间索引) 的分钟级 K 线热缓存。

    - md:{SYMBOL}:data    Hash, field=ts, value=JSON
    - md:{SYMBOL}:zindex  Sorted Set, score=ts, member=ts
    """

    def __init__(self, client: aioredis.Redis, retention_days: int):
        self._r = client
        self.retention_seconds = retention_days * 86400

    @classmethod
    async def connect(cls, url: str, retention_days: int = 7) -> "RedisStore":
        if url.startswith("fake://"):
            try:
                import fakeredis.aioredis as far
            except ImportError as e:
                raise RuntimeError(
                    "fake:// URL 需要安装 fakeredis (pip install fakeredis 或 uv sync --extra dev)"
                ) from e
            client = far.FakeRedis(decode_responses=True)
            logger.info(f"Redis 已连接 (fake): {url} (retention={retention_days}d)")
        else:
            client = aioredis.from_url(url, decode_responses=True)
            await client.ping()
            logger.info(f"Redis 已连接: {url} (retention={retention_days}d)")
        return cls(client, retention_days)

    async def close(self) -> None:
        await self._r.aclose()

    async def upsert(self, record: UnifiedMarketData) -> None:
        symbol = record.symbol.upper()
        ts = epoch_seconds(record)
        cutoff = ts - self.retention_seconds
        zkey = _zindex_key(symbol)
        dkey = _data_key(symbol)

        # 1. 找出超龄的成员（要从 Hash 一并删除）
        expired = await self._r.zrangebyscore(zkey, "-inf", f"({cutoff}")

        async with self._r.pipeline(transaction=True) as pipe:
            pipe.hset(dkey, str(ts), to_json(record))
            pipe.zadd(zkey, {str(ts): ts})
            if expired:
                pipe.hdel(dkey, *expired)
                pipe.zremrangebyscore(zkey, "-inf", f"({cutoff}")
            await pipe.execute()

    async def bulk_load(self, records: Iterable[UnifiedMarketData]) -> int:
        records = list(records)
        if not records:
            return 0

        by_symbol: dict[str, list[UnifiedMarketData]] = {}
        for r in records:
            by_symbol.setdefault(r.symbol.upper(), []).append(r)

        total = 0
        for symbol, group in by_symbol.items():
            zkey = _zindex_key(symbol)
            dkey = _data_key(symbol)
            mapping = {str(epoch_seconds(r)): to_json(r) for r in group}
            zmap = {str(epoch_seconds(r)): epoch_seconds(r) for r in group}

            # 以最新一条为基准计算 cutoff，做一次 trim
            latest_ts = max(zmap.values())
            cutoff = latest_ts - self.retention_seconds
            expired = await self._r.zrangebyscore(zkey, "-inf", f"({cutoff}")

            async with self._r.pipeline(transaction=True) as pipe:
                pipe.hset(dkey, mapping=mapping)
                pipe.zadd(zkey, zmap)
                if expired:
                    pipe.hdel(dkey, *expired)
                    pipe.zremrangebyscore(zkey, "-inf", f"({cutoff}")
                await pipe.execute()
            total += len(group)
        return total

    async def range(
        self, symbol: str, start_ts: int, end_ts: int
    ) -> list[UnifiedMarketData]:
        zkey = _zindex_key(symbol)
        dkey = _data_key(symbol)
        ts_list = await self._r.zrangebyscore(zkey, start_ts, end_ts)
        if not ts_list:
            return []
        blobs = await self._r.hmget(dkey, ts_list)
        return [from_json(b) for b in blobs if b is not None]

    async def count(self, symbol: str) -> int:
        return await self._r.zcard(_zindex_key(symbol))
