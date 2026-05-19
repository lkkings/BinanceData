"""SQLite 持久化存储（aiosqlite）"""
import logging
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Iterable

import aiosqlite

from ...domain import UnifiedMarketData
from .serialize import epoch_seconds, from_json, to_json

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS market_data_1m (
  symbol    TEXT    NOT NULL,
  timestamp INTEGER NOT NULL,
  data      TEXT    NOT NULL,
  PRIMARY KEY (symbol, timestamp)
);
"""

_INDEX = """
CREATE INDEX IF NOT EXISTS idx_md_symbol_ts
  ON market_data_1m(symbol, timestamp);
"""


class SQLiteStore:
    """1 分钟聚合数据的本地真源存储"""

    def __init__(self, db: aiosqlite.Connection, path: Path):
        self._db = db
        self.path = path

    @classmethod
    async def connect(cls, path: Path) -> "SQLiteStore":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        db = await aiosqlite.connect(path)
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.executescript(_SCHEMA + _INDEX)
        await db.commit()
        logger.info(f"SQLite 已连接: {path}")
        return cls(db, path)

    async def close(self) -> None:
        await self._db.close()

    async def upsert(self, record: UnifiedMarketData) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO market_data_1m(symbol, timestamp, data) VALUES (?, ?, ?)",
            (record.symbol.upper(), epoch_seconds(record), to_json(record)),
        )
        await self._db.commit()

    async def upsert_many(self, records: Iterable[UnifiedMarketData]) -> int:
        rows = [
            (r.symbol.upper(), epoch_seconds(r), to_json(r))
            for r in records
        ]
        if not rows:
            return 0
        await self._db.executemany(
            "INSERT OR REPLACE INTO market_data_1m(symbol, timestamp, data) VALUES (?, ?, ?)",
            rows,
        )
        await self._db.commit()
        return len(rows)

    async def range(
        self, symbol: str, start_ts: int, end_ts: int
    ) -> list[UnifiedMarketData]:
        async with self._db.execute(
            "SELECT data FROM market_data_1m "
            "WHERE symbol = ? AND timestamp >= ? AND timestamp <= ? "
            "ORDER BY timestamp",
            (symbol.upper(), start_ts, end_ts),
        ) as cursor:
            rows = await cursor.fetchall()
        return [from_json(row[0]) for row in rows]

    async def has_full_day(self, symbol: str, day: date, min_records: int = 1400) -> bool:
        """该天 SQLite 中是否已经有足够的分钟级记录（默认 1400/1440）"""
        return (await self.count_day(symbol, day)) >= min_records

    async def count_day(self, symbol: str, day: date) -> int:
        """返回某天的分钟数据条数"""
        start = int(datetime.combine(day, time.min, tzinfo=timezone.utc).timestamp())
        end = start + 86400 - 1
        async with self._db.execute(
            "SELECT COUNT(*) FROM market_data_1m "
            "WHERE symbol = ? AND timestamp BETWEEN ? AND ?",
            (symbol.upper(), start, end),
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0
