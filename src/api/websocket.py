"""WebSocket 订阅端点

协议（JSON-only，UTF-8）：

客户端 → 服务：
    {"action": "subscribe", "symbols": ["btcusdt"], "channels": ["minute"], "history_minutes": 60}
    {"action": "unsubscribe", "symbols": ["ethusdt"]}
    {"action": "ping"}

服务 → 客户端：
    {"type": "subscribed", "symbols": [...], "channels": [...]}
    {"type": "backfill", "symbol": "btcusdt", "channel": "minute", "records": [...]}
    {"type": "data", "symbol": "btcusdt", "channel": "minute", "record": {...}}
    {"type": "pong"}
    {"type": "error", "message": "..."}

实现要点：所有客户端方向的写入都通过 _send 串行化，避免 receive 协程和 pump 协程
并发调用 websocket.send_json 触发竞争。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from ..core.pubsub import Subscription
from ..infrastructure.storage.serialize import to_payload_dict
from ..logging import get_logger
from .deps import Container

logger = get_logger(__name__)
router = APIRouter()

SUPPORTED_CHANNELS = {"minute"}


def _err(message: str, **extra: Any) -> dict[str, Any]:
    return {"type": "error", "message": message, **extra}


def _clamp_history(value: Any, default: int, maximum: int) -> int:
    try:
        v = int(value) if value is not None else default
    except (TypeError, ValueError):
        v = default
    return max(0, min(v, maximum))


async def _validate_token(ws: WebSocket, container: Container, token: str | None) -> bool:
    expected = container.settings.api_key
    if not expected:
        return True
    if token and token == expected:
        return True
    await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="unauthorized")
    return False


async def _backfill_records(container: Container, symbol: str, history_minutes: int) -> list[dict]:
    if history_minutes <= 0:
        return []
    now = datetime.now(tz=timezone.utc)
    end_ts = int(now.timestamp())
    start_ts = end_ts - history_minutes * 60

    # 先尝试 Redis
    try:
        records = await container.redis_store.range(symbol, start_ts, end_ts)
    except Exception as e:
        logger.warning("ws.backfill.redis_fail", symbol=symbol, error=str(e))
        records = []

    # 不足时查 SQLite
    if len(records) < history_minutes:
        try:
            records = await container.sqlite_store.range(symbol.upper(), start_ts, end_ts)
        except Exception as e:
            logger.warning("ws.backfill.sqlite_fail", symbol=symbol, error=str(e))

    # 仍然不足时按需从历史数据拉取
    if len(records) < history_minutes:
        try:
            await container.on_demand_fetcher.ensure_range(symbol, start_ts, end_ts)
            records = await container.sqlite_store.range(symbol.upper(), start_ts, end_ts)
        except Exception as e:
            logger.warning("ws.backfill.on_demand_fail", symbol=symbol, error=str(e))

    return [to_payload_dict(r) for r in records]


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str | None = Query(default=None),
) -> None:
    container: Container = websocket.app.state.container
    settings = container.settings

    await websocket.accept()
    if not await _validate_token(websocket, container, token):
        return

    sub: Subscription | None = None
    pump_task: asyncio.Task[None] | None = None
    subscribed_symbols: set[str] = set()
    send_lock = asyncio.Lock()

    async def safe_send(payload: dict) -> bool:
        async with send_lock:
            try:
                await websocket.send_json(payload)
                return True
            except Exception:
                return False

    async def pump_loop() -> None:
        assert sub is not None
        while not sub.closed:
            try:
                msg = await sub.queue.get()
            except asyncio.CancelledError:
                return
            ok = await safe_send({
                "type": "data",
                "symbol": msg.symbol,
                "channel": msg.channel,
                "record": msg.payload,
            })
            if not ok:
                return

    async def cleanup() -> None:
        nonlocal pump_task, sub
        if pump_task and not pump_task.done():
            pump_task.cancel()
            try:
                await pump_task
            except (asyncio.CancelledError, Exception):
                pass
        if sub is not None:
            container.pubsub.unsubscribe(sub)
            sub = None

    try:
        while True:
            try:
                raw = await websocket.receive_json()
            except WebSocketDisconnect:
                break
            except Exception as e:
                await safe_send(_err(f"invalid_json: {e}"))
                continue

            action = (raw.get("action") or "").lower()

            if action == "ping":
                await safe_send({"type": "pong"})
                continue

            if action == "subscribe":
                channels = {c.lower() for c in raw.get("channels") or ["minute"]}
                bad = channels - SUPPORTED_CHANNELS
                if bad:
                    await safe_send(_err(f"unsupported_channels: {sorted(bad)}"))
                    continue

                req_symbols = {s.lower() for s in raw.get("symbols") or []}
                if not req_symbols:
                    await safe_send(_err("symbols_required"))
                    continue

                unknown = req_symbols - set(container.pipeline.symbols)
                if unknown:
                    await safe_send(
                        _err(
                            "unknown_symbols",
                            unknown=sorted(unknown),
                            available=container.pipeline.symbols,
                        )
                    )
                    continue

                history_minutes = _clamp_history(
                    raw.get("history_minutes"),
                    default=settings.ws_default_history_minutes,
                    maximum=settings.ws_max_history_minutes,
                )

                # 先准备好 backfill（异步 IO 可能耗时）
                backfills: list[tuple[str, list[dict]]] = []
                for symbol in sorted(req_symbols):
                    records = await _backfill_records(container, symbol, history_minutes)
                    backfills.append((symbol, records))

                # 然后建立订阅 + 启动 pump，确保订阅期间的实时消息会进队列
                if sub is None:
                    sub = container.pubsub.subscribe(
                        channels=channels,
                        symbols=req_symbols,
                    )
                    pump_task = asyncio.create_task(pump_loop(), name="ws-pump")
                else:
                    container.pubsub.update_subscription(
                        sub, add_symbols=req_symbols, add_channels=channels
                    )

                subscribed_symbols |= req_symbols

                await safe_send({
                    "type": "subscribed",
                    "symbols": sorted(subscribed_symbols),
                    "channels": sorted(sub.channels),
                    "history_minutes": history_minutes,
                })
                for symbol, records in backfills:
                    await safe_send({
                        "type": "backfill",
                        "symbol": symbol,
                        "channel": "minute",
                        "records": records,
                    })
                continue

            if action == "unsubscribe":
                if sub is None:
                    await safe_send(_err("not_subscribed"))
                    continue
                req_symbols = {s.lower() for s in raw.get("symbols") or []}
                req_channels = {c.lower() for c in raw.get("channels") or []}
                container.pubsub.update_subscription(
                    sub,
                    remove_symbols=req_symbols or None,
                    remove_channels=req_channels or None,
                )
                subscribed_symbols -= req_symbols
                await safe_send({
                    "type": "unsubscribed",
                    "symbols": sorted(subscribed_symbols),
                    "channels": sorted(sub.channels),
                })
                continue

            await safe_send(_err(f"unknown_action: {action}"))

    finally:
        await cleanup()
