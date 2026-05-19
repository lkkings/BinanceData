"""WebSocket 客户端示例

用法:
    uv run python examples/ws_client.py                       # 默认 ws://127.0.0.1:8000/ws，订阅 btcusdt
    uv run python examples/ws_client.py --url ws://host/ws --symbols btcusdt,ethusdt --history 60
    uv run python examples/ws_client.py --token <API_KEY>     # 启用了 API_KEY 时

所有时间戳以 UTC+8 (Asia/Shanghai) 显示。
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Any

import websockets

CST = timezone(timedelta(hours=8))


def _ts_to_cst(ts: Any) -> str:
    """timestamp 已是 UTC+8 字符串（如 '2026-05-19 13:48:00'），直接返回；兼容老的 epoch 秒。"""
    if isinstance(ts, str):
        return ts
    try:
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(CST)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return str(ts)


def _now_cst() -> str:
    return datetime.now(tz=CST).strftime("%Y-%m-%d %H:%M:%S.%f CST")[:-3]


def _format_record(rec: dict) -> str:
    """把单条 UnifiedMarketData JSON 渲染成多行可读文本。"""
    ts_cst = _ts_to_cst(rec.get("timestamp"))
    lines = [f"  时间(UTC+8): {ts_cst}    交易对: {rec.get('symbol')}"]

    def _g(*keys: str) -> str:
        parts = []
        for k in keys:
            v = rec.get(k)
            parts.append(f"{k}={v}" if v is not None else f"{k}=-")
        return "  ".join(parts)

    lines.append("  OHLCV  " + _g("open", "high", "low", "close", "volume", "quote_volume", "vwap"))
    lines.append("  成交  " + _g("trade_count", "buy_count", "sell_count", "buy_volume", "sell_volume"))
    lines.append("  高频  " + _g("trade_intensity", "avg_trade_size", "max_trade_size", "price_range"))
    lines.append("  Tick  " + _g("tick_count", "up_tick_count", "down_tick_count", "volume_imbalance"))
    lines.append("  大单  " + _g("large_trade_count", "large_trade_volume"))
    lines.append(
        "  Top   "
        + _g("best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty", "spread_bps", "mid_price", "imbalance_5")
    )
    lines.append(
        "  深度  "
        + _g(
            "bid_depth_02", "ask_depth_02", "depth_imbalance_02",
            "bid_depth_1", "ask_depth_1", "depth_imbalance_1",
            "total_bid_depth", "total_ask_depth", "depth_imbalance_total",
        )
    )
    lines.append(
        "  K线   "
        + _g(
            "kline_open", "kline_high", "kline_low", "kline_close",
            "kline_volume", "kline_quote_volume", "kline_count",
            "kline_taker_buy_ratio", "kline_body_ratio",
            "kline_upper_shadow", "kline_lower_shadow",
        )
    )
    lines.append(f"  update_count={rec.get('update_count')}")
    return "\n".join(lines)


async def run(url: str, symbols: list[str], history_minutes: int, token: str | None) -> None:
    final_url = url
    if token:
        sep = "&" if "?" in url else "?"
        final_url = f"{url}{sep}token={token}"

    print(f"[{_now_cst()}] 连接 {final_url}")
    async with websockets.connect(final_url) as ws:
        sub = {
            "action": "subscribe",
            "symbols": symbols,
            "channels": ["minute"],
            "history_minutes": history_minutes,
        }
        print(f"[{_now_cst()}] 发送订阅: {sub}")
        await ws.send(json.dumps(sub))

        # 周期性 ping，保活
        async def ping_loop():
            while True:
                await asyncio.sleep(30)
                try:
                    await ws.send(json.dumps({"action": "ping"}))
                except Exception:
                    return

        ping_task = asyncio.create_task(ping_loop())

        try:
            async for raw in ws:
                msg = json.loads(raw)
                t = msg.get("type")
                now = _now_cst()

                if t == "subscribed":
                    print(f"\n[{now}] [SUBSCRIBED]")
                    print(f"  symbols={msg['symbols']}  channels={msg['channels']}  history_minutes={msg.get('history_minutes')}")
                elif t == "backfill":
                    records = msg.get("records") or []
                    print(f"\n[{now}] [BACKFILL] symbol={msg['symbol']}  count={len(records)}")
                    if records:
                        first = _ts_to_cst(records[0]["timestamp"])
                        last = _ts_to_cst(records[-1]["timestamp"])
                        print(f"  范围(UTC+8): {first}  ~  {last}")
                        # 只详细打印最后一条避免刷屏
                        print("  最后一条:")
                        print(_format_record(records[-1]))
                elif t == "data":
                    print(f"\n[{now}] [DATA] symbol={msg['symbol']}  channel={msg['channel']}")
                    print(_format_record(msg["record"]))
                elif t == "pong":
                    print(f"[{now}] [PONG]")
                elif t == "unsubscribed":
                    print(f"[{now}] [UNSUBSCRIBED] {msg}")
                elif t == "error":
                    print(f"[{now}] [ERROR] {msg.get('message')}  extra={ {k: v for k, v in msg.items() if k not in ('type','message')} }")
                else:
                    print(f"[{now}] [UNKNOWN] {msg}")
        finally:
            ping_task.cancel()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BinanceData WebSocket 客户端示例 (时间显示 UTC+8)")
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    p.add_argument("--symbols", default="btcusdt", help="逗号分隔，例如 btcusdt,ethusdt")
    p.add_argument("--history", type=int, default=10, help="订阅时回放的历史分钟数")
    p.add_argument("--token", default=None, help="如服务端启用 API_KEY 则需提供")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [s.strip().lower() for s in args.symbols.split(",") if s.strip()]
    try:
        asyncio.run(run(args.url, symbols, args.history, args.token))
    except KeyboardInterrupt:
        print("\n[client] interrupted")


if __name__ == "__main__":
    main()
