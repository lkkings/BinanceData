"""WebSocket API 集成测试

为了避免跨事件循环发布导致 TestClient 死锁，这里只测：
- HTTP 端点
- 订阅 → backfill（通过预先写入 SQLite 来验证回放路径）
- 协议错误处理
- ping/pong、unsubscribe 协议响应
- API key 鉴权

实时推送链路（Pipeline → PubSub → 订阅者队列）由 tests/test_pipeline.py 直接测。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import get_settings
from src.domain import UnifiedMarketData


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SYMBOLS", "btcusdt,ethusdt")
    monkeypatch.setenv("REDIS_URL", "fake://")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "ws_api.db"))
    monkeypatch.setenv("HISTORY_BACKFILL_DAYS", "0")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("API_KEY", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _patched_app():
    fake_collector = AsyncMock()
    fake_collector.start = AsyncMock()
    fake_collector.stop = AsyncMock()
    return patch("src.core.pipeline.BinanceCollector", return_value=fake_collector)


def _make_record(symbol: str, minute_offset: int = 0) -> UnifiedMarketData:
    base = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
    return UnifiedMarketData(
        timestamp=base + timedelta(minutes=minute_offset),
        symbol=symbol.upper(),
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100.5"),
        volume=Decimal("1.5"),
    )


def test_healthz_and_symbols_endpoints():
    with _patched_app():
        app = create_app()
        with TestClient(app) as client:
            r = client.get("/healthz")
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "ok"
            assert set(body["symbols"]) == {"btcusdt", "ethusdt"}

            r2 = client.get("/symbols")
            assert r2.status_code == 200
            assert r2.json() == {
                "symbols": ["btcusdt", "ethusdt"],
                "channels": ["minute"],
            }


def test_subscribe_returns_backfill_from_sqlite():
    """预先把数据写入 SQLite，订阅时应通过 backfill 推回。"""
    with _patched_app():
        app = create_app()
        with TestClient(app) as client:
            # 通过 portal 在应用循环内 seed 数据
            container = app.state.container
            record = _make_record("BTCUSDT")

            client.portal.call(container.sqlite_store.upsert, record)

            with client.websocket_connect("/ws") as ws:
                ws.send_json({
                    "action": "subscribe",
                    "symbols": ["btcusdt"],
                    "channels": ["minute"],
                    "history_minutes": 5,
                })

                resp = ws.receive_json()
                assert resp["type"] == "subscribed"
                assert resp["symbols"] == ["btcusdt"]

                bf = ws.receive_json()
                assert bf["type"] == "backfill"
                assert bf["symbol"] == "btcusdt"
                assert len(bf["records"]) == 1
                assert bf["records"][0]["close"] == "100.5"


def test_subscribe_with_zero_history_yields_empty_backfill():
    with _patched_app():
        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({
                    "action": "subscribe",
                    "symbols": ["btcusdt"],
                    "channels": ["minute"],
                    "history_minutes": 0,
                })
                assert ws.receive_json()["type"] == "subscribed"
                bf = ws.receive_json()
                assert bf["type"] == "backfill"
                assert bf["records"] == []


def test_unknown_symbol_rejected():
    with _patched_app():
        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({
                    "action": "subscribe",
                    "symbols": ["dogeusdt"],
                    "channels": ["minute"],
                })
                resp = ws.receive_json()
                assert resp["type"] == "error"
                assert "unknown" in resp["message"]


def test_unsupported_channel_rejected():
    with _patched_app():
        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({
                    "action": "subscribe",
                    "symbols": ["btcusdt"],
                    "channels": ["trade"],
                })
                resp = ws.receive_json()
                assert resp["type"] == "error"
                assert "unsupported_channels" in resp["message"]


def test_ping_pong():
    with _patched_app():
        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"action": "ping"})
                assert ws.receive_json() == {"type": "pong"}


def test_unsubscribe_updates_symbol_set():
    with _patched_app():
        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({
                    "action": "subscribe",
                    "symbols": ["btcusdt", "ethusdt"],
                    "channels": ["minute"],
                    "history_minutes": 0,
                })
                # subscribed + 2 backfill
                assert ws.receive_json()["type"] == "subscribed"
                assert ws.receive_json()["type"] == "backfill"
                assert ws.receive_json()["type"] == "backfill"

                ws.send_json({"action": "unsubscribe", "symbols": ["btcusdt"]})
                resp = ws.receive_json()
                assert resp["type"] == "unsubscribed"
                assert resp["symbols"] == ["ethusdt"]


def test_api_key_required_when_set(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-token")
    get_settings.cache_clear()

    with _patched_app():
        app = create_app()
        with TestClient(app) as client:
            with pytest.raises(Exception):
                with client.websocket_connect("/ws") as ws:
                    ws.receive_json()

            with client.websocket_connect("/ws?token=secret-token") as ws:
                ws.send_json({"action": "ping"})
                assert ws.receive_json() == {"type": "pong"}
