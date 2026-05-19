"""历史预热 + 实时订阅的组合工作流

典型策略 / 行情面板启动流程：
1. 通过 HTTP /api/range 拉取最近 N 分钟历史数据进行预热（构造滑动窗口）
2. 切到 WebSocket /ws 订阅实时分钟数据
3. 实时数据到达时滑动窗口前进，重新计算指标（这里演示简单 EMA 与 1 分钟收益率）

用法:
    uv run python examples/live_with_warmup.py --symbol btcusdt --warmup 60
    uv run python examples/live_with_warmup.py --symbol btcusdt --warmup 120 --ema-period 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import deque
from datetime import datetime, timedelta, timezone

import httpx
import websockets


CST = timezone(timedelta(hours=8))


def _now_cst_str() -> str:
    return datetime.now(tz=CST).strftime("%Y-%m-%d %H:%M:%S")


def _ts_to_cst_str(ts) -> str:
    if isinstance(ts, str):
        return ts
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(CST).strftime("%Y-%m-%d %H:%M:%S")


class RollingEMA:
    """简单 EMA 指标（滑动 window 不需要，但 EMA 需要历史种子）"""

    def __init__(self, period: int):
        self.period = period
        self.alpha = 2 / (period + 1)
        self._value: float | None = None

    def update(self, price: float) -> float:
        if self._value is None:
            self._value = price
        else:
            self._value = self.alpha * price + (1 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> float | None:
        return self._value


async def warmup_via_http(
    base_url: str, symbol: str, warmup_minutes: int
) -> list[dict]:
    """从 /api/range 拉取最近 warmup_minutes 分钟的历史数据"""
    now = datetime.now(tz=CST)
    start = (now - timedelta(minutes=warmup_minutes)).strftime("%Y-%m-%d %H:%M:%S")
    end = now.strftime("%Y-%m-%d %H:%M:%S")

    print(f"[{_now_cst_str()}] [WARMUP] fetching {warmup_minutes} 分钟历史 ({start} ~ {end})")

    rows: list[dict] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        async with client.stream(
            "GET",
            f"{base_url}/api/range",
            params={"symbol": symbol, "start": start, "end": end},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                if isinstance(msg, dict) and "records" in msg:
                    rows.extend(msg["records"])
                else:
                    rows.append(msg)
    print(f"[{_now_cst_str()}] [WARMUP] 加载 {len(rows)} 条历史记录")
    return rows


async def subscribe_live(
    ws_url: str, symbol: str, on_record: callable
) -> None:
    """订阅 WebSocket 实时分钟数据，回调 on_record(record_dict)"""
    print(f"[{_now_cst_str()}] [LIVE] connecting {ws_url}")
    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({
            "action": "subscribe",
            "symbols": [symbol],
            "channels": ["minute"],
            "history_minutes": 0,  # 不需要后端再推历史
        }))

        async for raw in ws:
            msg = json.loads(raw)
            t = msg.get("type")
            if t == "subscribed":
                print(f"[{_now_cst_str()}] [LIVE] subscribed: {msg['symbols']}")
            elif t == "data":
                on_record(msg["record"])
            elif t == "error":
                print(f"[{_now_cst_str()}] [ERROR] {msg.get('message')}")


async def run(symbol: str, http_url: str, ws_url: str, warmup: int, ema_period: int) -> None:
    # 预热阶段：用历史数据初始化指标
    historical = await warmup_via_http(http_url, symbol, warmup)

    window: deque = deque(maxlen=warmup)
    ema = RollingEMA(period=ema_period)

    for r in historical:
        close = float(r["close"])
        window.append(close)
        ema.update(close)

    last_close = window[-1] if window else None
    ema_str = f"{ema.value:.2f}" if ema.value is not None else "n/a"
    print(f"[{_now_cst_str()}] [WARMUP] 完成  最新close={last_close}  EMA{ema_period}={ema_str}")
    print(f"[{_now_cst_str()}] [LIVE] 切换到实时流...")
    print()

    # 实时阶段
    def on_record(rec: dict) -> None:
        ts = rec["timestamp"]
        close = float(rec["close"])
        volume = float(rec["volume"])

        window.append(close)
        ema.update(close)

        ret = (close / window[-2] - 1) * 100 if len(window) >= 2 else 0.0
        signal = ""
        if ema.value is not None:
            if close > ema.value:
                signal = "↑ above EMA"
            elif close < ema.value:
                signal = "↓ below EMA"

        print(
            f"[{_now_cst_str()}] [DATA] {ts}  "
            f"close={close:.2f}  ret={ret:+.3f}%  "
            f"vol={volume:.4f}  EMA{ema_period}={ema.value:.2f}  {signal}"
        )

    await subscribe_live(ws_url, symbol, on_record)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="历史预热 + 实时订阅的组合工作流")
    p.add_argument("--http-url", default="http://127.0.0.1:8000")
    p.add_argument("--ws-url", default="ws://127.0.0.1:8000/ws")
    p.add_argument("--symbol", default="btcusdt")
    p.add_argument("--warmup", type=int, default=60, help="预热分钟数（默认 60）")
    p.add_argument("--ema-period", type=int, default=20, help="EMA 周期（默认 20）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(args.symbol, args.http_url, args.ws_url, args.warmup, args.ema_period))
    except KeyboardInterrupt:
        print(f"\n[{_now_cst_str()}] interrupted")


if __name__ == "__main__":
    main()
