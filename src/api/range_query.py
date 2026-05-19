"""HTTP 区间查询接口（NDJSON 流式）

GET /api/range?symbol=btcusdt&start=2026-05-10&end=2026-05-15

参数：
    symbol     必填，交易对（小写或大写均可）
    start      必填，UTC+8 时间，支持 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS'
    end        必填，UTC+8 时间，同 start 格式
    chunk_size 可选，每个 JSON 块包含的记录数（默认 1，逐条流式）

响应：
    Content-Type: application/x-ndjson
    每行一个 UnifiedMarketData JSON 对象（timestamp 为 UTC+8 字符串）。
    缺失的日期会先从 Binance 历史数据自动拉取后再返回。
"""
from __future__ import annotations

import json
from datetime import datetime, time, timedelta, timezone
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from ..infrastructure.storage.serialize import to_payload_dict
from ..logging import get_logger
from .deps import Container

logger = get_logger(__name__)
router = APIRouter()

CST = timezone(timedelta(hours=8))


def _parse_cst(value: str, *, end_of_day: bool) -> datetime:
    """解析 UTC+8 时间字符串为 UTC datetime。

    支持 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS'。
    end_of_day=True 时，纯日期会被解释为当天 23:59:59。
    """
    value = value.strip()
    fmts = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d")
    for fmt in fmts:
        try:
            dt = datetime.strptime(value, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"无法解析时间: {value!r}（期望 YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS）")

    if fmt == "%Y-%m-%d" and end_of_day:
        dt = datetime.combine(dt.date(), time(23, 59, 59))

    return dt.replace(tzinfo=CST).astimezone(timezone.utc)


@router.get("/api/range")
async def stream_range(
    request: Request,
    symbol: str = Query(..., description="交易对，例如 btcusdt"),
    start: str = Query(..., description="起始时间（UTC+8），YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS"),
    end: str = Query(..., description="结束时间（UTC+8），同 start 格式"),
    chunk_size: int = Query(1, ge=1, le=1000, description="每个流式块的记录数"),
) -> StreamingResponse:
    container: Container = request.app.state.container

    # 解析时间范围
    try:
        start_dt = _parse_cst(start, end_of_day=False)
        end_dt = _parse_cst(end, end_of_day=True)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    if start_dt >= end_dt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start 必须早于 end",
        )

    symbol_upper = symbol.upper()
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    logger.info(
        "range.request",
        symbol=symbol_upper,
        start=start_dt.isoformat(),
        end=end_dt.isoformat(),
    )

    async def gen() -> AsyncIterator[bytes]:
        # 1) 自动拉取缺失日期
        try:
            await container.on_demand_fetcher.ensure_range(symbol_upper, start_ts, end_ts)
        except Exception as e:
            logger.error("range.fetch_failed", error=str(e), symbol=symbol_upper)

        # 2) 从 SQLite 拉取范围内全部记录
        records = await container.sqlite_store.range(symbol_upper, start_ts, end_ts)
        if not records:
            return

        # 3) NDJSON 流式输出
        if chunk_size == 1:
            for r in records:
                yield (json.dumps(to_payload_dict(r), separators=(",", ":")) + "\n").encode("utf-8")
        else:
            for i in range(0, len(records), chunk_size):
                batch = records[i : i + chunk_size]
                payload = {"records": [to_payload_dict(r) for r in batch]}
                yield (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson",
        headers={
            "X-Symbol": symbol_upper,
            "X-Start-CST": start_dt.astimezone(CST).strftime("%Y-%m-%d %H:%M:%S"),
            "X-End-CST": end_dt.astimezone(CST).strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
