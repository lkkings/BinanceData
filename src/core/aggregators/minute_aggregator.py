"""分钟级实时流式聚合器

直接从 trade / depth / kline WebSocket 流，按 1 分钟分桶计算 UnifiedMarketData。
不依赖任何秒级中间表示。
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable, Optional

import pandas as pd

from ...domain.aggregated import UnifiedMarketData

logger = logging.getLogger(__name__)


# 与 history bookDepth 对齐的档位口径
DEPTH_PCT_02 = Decimal("0.002")  # ±0.2%
DEPTH_PCT_1 = Decimal("0.01")    # ±1%


class MinuteAggregator:
    """分钟级数据聚合器

    把实时 trade / depth / kline 流以分钟为单位聚合，每分钟产出一条 UnifiedMarketData。
    """

    def __init__(
        self,
        on_aggregated_data: Callable[[UnifiedMarketData], None] | None = None,
        watermark_delay_seconds: int = 5,
    ):
        self.on_aggregated_data = on_aggregated_data

        # 数据缓冲区（按分钟分组）
        self.orderbook_buffer: dict[int, list[dict]] = defaultdict(list)
        self.trade_buffer: dict[int, list[dict]] = defaultdict(list)

        # 当前订单簿状态（用于计算分钟末快照特征）
        self.current_orderbook: dict[str, dict] = {}

        # 上一分钟的收盘价（用于零成交分钟填充）
        self.last_close: dict[str, Decimal] = {}

        # 事件时间水位线（最大已见事件分钟）
        self.max_event_minute = 0
        self.last_flushed_minute = 0
        self.watermark_delay_seconds = watermark_delay_seconds

        self.aggregation_task: asyncio.Task | None = None
        self.running = False

    async def start(self) -> None:
        if not self.running:
            self.running = True
            self.aggregation_task = asyncio.create_task(self._aggregation_loop())
            logger.info("分钟级聚合器已启动")

    async def stop(self) -> None:
        self.running = False
        if self.aggregation_task:
            self.aggregation_task.cancel()
            try:
                await self.aggregation_task
            except asyncio.CancelledError:
                pass
        logger.info("分钟级聚合器已停止")

    @staticmethod
    def _minute_of(timestamp_ms: int) -> int:
        return timestamp_ms // 60000

    def add_orderbook_update(self, symbol: str, data: dict) -> None:
        minute = self._minute_of(data["event_time"])
        self.orderbook_buffer[minute].append({"symbol": symbol, "data": data})
        self._update_orderbook_state(symbol, data)
        if minute > self.max_event_minute:
            self.max_event_minute = minute

    def add_trade(self, symbol: str, data: dict) -> None:
        minute = self._minute_of(data["trade_time"])
        self.trade_buffer[minute].append({"symbol": symbol, "data": data})
        if minute > self.max_event_minute:
            self.max_event_minute = minute

    def _update_orderbook_state(self, symbol: str, data: dict) -> None:
        if symbol not in self.current_orderbook:
            self.current_orderbook[symbol] = {"bids": {}, "asks": {}}
        book = self.current_orderbook[symbol]
        for price, qty in data["bids"]:
            if qty == 0:
                book["bids"].pop(price, None)
            else:
                book["bids"][price] = qty
        for price, qty in data["asks"]:
            if qty == 0:
                book["asks"].pop(price, None)
            else:
                book["asks"][price] = qty

    async def _aggregation_loop(self) -> None:
        delay_minutes = max(1, self.watermark_delay_seconds // 60 + 1)
        while self.running:
            try:
                await asyncio.sleep(1.0)

                if self.max_event_minute == 0:
                    continue

                if self.last_flushed_minute == 0:
                    self.last_flushed_minute = self.max_event_minute - delay_minutes - 1

                flushable_minute = self.max_event_minute - delay_minutes

                for minute in range(self.last_flushed_minute + 1, flushable_minute + 1):
                    await self._aggregate_minute(minute)
                    self.orderbook_buffer.pop(minute, None)
                    self.trade_buffer.pop(minute, None)
                    self.last_flushed_minute = minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"分钟聚合循环错误: {e}", exc_info=True)

    async def _aggregate_minute(self, minute: int) -> None:
        ob_by_symbol: dict[str, list[dict]] = defaultdict(list)
        for item in self.orderbook_buffer.get(minute, []):
            ob_by_symbol[item["symbol"]].append(item["data"])

        tr_by_symbol: dict[str, list[dict]] = defaultdict(list)
        for item in self.trade_buffer.get(minute, []):
            tr_by_symbol[item["symbol"]].append(item["data"])

        known_symbols = set(self.last_close.keys()) | set(self.current_orderbook.keys())
        all_symbols = set(ob_by_symbol.keys()) | set(tr_by_symbol.keys()) | known_symbols

        for symbol in all_symbols:
            try:
                ob_updates = ob_by_symbol.get(symbol, [])
                tr_list = tr_by_symbol.get(symbol, [])

                ob_feat = (
                    self._compute_orderbook_features(symbol, ob_updates)
                    if ob_updates else None
                )
                tr_feat = (
                    self._compute_trade_features(tr_list)
                    if tr_list else None
                )

                if tr_feat is None:
                    last = self.last_close.get(symbol)
                    if last is None:
                        continue
                    tr_feat = self._empty_trade_features(last)
                else:
                    self.last_close[symbol] = tr_feat["close"]

                unified = self._build_unified(minute, symbol, tr_feat, ob_feat)

                if self.on_aggregated_data:
                    self.on_aggregated_data(unified)

            except Exception as e:
                logger.error(f"分钟聚合错误 {symbol}: {e}", exc_info=True)

    @staticmethod
    def _build_unified(
        minute: int,
        symbol: str,
        tr_feat: dict,
        ob_feat: Optional[dict],
    ) -> UnifiedMarketData:
        # 从 trade 流重建 kline 字段（futures kline WS 不可用）
        open_p = tr_feat["open"]
        high_p = tr_feat["high"]
        low_p = tr_feat["low"]
        close_p = tr_feat["close"]
        volume = tr_feat["volume"]
        buy_volume = tr_feat["buy_volume"]
        buy_quote_volume = tr_feat["buy_quote_volume"]

        hl_range = high_p - low_p if high_p and low_p else Decimal("0")
        body = abs(close_p - open_p) if close_p and open_p else Decimal("0")

        kline_taker_buy_ratio = float(buy_volume / volume) if volume and volume > 0 else 0.5
        kline_body_ratio = float(body / hl_range) if hl_range > 0 else 0.0
        kline_upper_shadow = (
            float((high_p - max(open_p, close_p)) / hl_range) if hl_range > 0 else 0.0
        )
        kline_lower_shadow = (
            float((min(open_p, close_p) - low_p) / hl_range) if hl_range > 0 else 0.0
        )

        return UnifiedMarketData(
            timestamp=datetime.fromtimestamp(minute * 60, tz=timezone.utc),
            symbol=symbol,
            open=open_p, high=high_p, low=low_p, close=close_p,
            volume=volume, quote_volume=tr_feat["quote_volume"], vwap=tr_feat["vwap"],
            trade_count=tr_feat["trade_count"],
            buy_count=tr_feat["buy_count"], sell_count=tr_feat["sell_count"],
            buy_volume=buy_volume, sell_volume=tr_feat["sell_volume"],
            buy_quote_volume=buy_quote_volume, sell_quote_volume=tr_feat["sell_quote_volume"],
            trade_intensity=tr_feat["trade_intensity"],
            avg_trade_size=tr_feat["avg_trade_size"], max_trade_size=tr_feat["max_trade_size"],
            price_range=tr_feat["price_range"],
            tick_count=tr_feat["tick_count"],
            up_tick_count=tr_feat["up_tick_count"], down_tick_count=tr_feat["down_tick_count"],
            volume_imbalance=tr_feat["volume_imbalance"],
            large_trade_count=tr_feat["large_trade_count"],
            large_trade_volume=tr_feat["large_trade_volume"],
            best_bid_price=ob_feat["best_bid_price"] if ob_feat else None,
            best_bid_qty=ob_feat["best_bid_qty"] if ob_feat else None,
            best_ask_price=ob_feat["best_ask_price"] if ob_feat else None,
            best_ask_qty=ob_feat["best_ask_qty"] if ob_feat else None,
            spread_bps=ob_feat["spread_bps"] if ob_feat else None,
            mid_price=ob_feat["mid_price"] if ob_feat else None,
            imbalance_5=ob_feat["imbalance_5"] if ob_feat else None,
            update_count=ob_feat["update_count"] if ob_feat else 0,
            bid_depth_02=ob_feat["bid_depth_02"] if ob_feat else None,
            ask_depth_02=ob_feat["ask_depth_02"] if ob_feat else None,
            bid_notional_02=ob_feat["bid_notional_02"] if ob_feat else None,
            ask_notional_02=ob_feat["ask_notional_02"] if ob_feat else None,
            depth_imbalance_02=ob_feat["depth_imbalance_02"] if ob_feat else None,
            bid_depth_1=ob_feat["bid_depth_1"] if ob_feat else None,
            ask_depth_1=ob_feat["ask_depth_1"] if ob_feat else None,
            depth_imbalance_1=ob_feat["depth_imbalance_1"] if ob_feat else None,
            total_bid_depth=ob_feat["total_bid_depth"] if ob_feat else None,
            total_ask_depth=ob_feat["total_ask_depth"] if ob_feat else None,
            depth_imbalance_total=ob_feat["depth_imbalance_total"] if ob_feat else None,
            kline_open=float(open_p) if open_p else None,
            kline_high=float(high_p) if high_p else None,
            kline_low=float(low_p) if low_p else None,
            kline_close=float(close_p) if close_p else None,
            kline_volume=float(volume) if volume else None,
            kline_quote_volume=float(tr_feat["quote_volume"]) if tr_feat["quote_volume"] else None,
            kline_count=tr_feat["trade_count"],
            kline_taker_buy_volume=float(buy_volume) if buy_volume else None,
            kline_taker_buy_quote_volume=float(buy_quote_volume) if buy_quote_volume else None,
            kline_taker_buy_ratio=kline_taker_buy_ratio,
            kline_body_ratio=kline_body_ratio,
            kline_upper_shadow=kline_upper_shadow,
            kline_lower_shadow=kline_lower_shadow,
        )

    # === 特征计算（无状态，纯函数；之前位于 SecondAggregator） ===

    @staticmethod
    def _empty_trade_features(last_close: Decimal) -> dict:
        zero = Decimal("0")
        return {
            "open": last_close, "high": last_close, "low": last_close, "close": last_close,
            "volume": zero, "quote_volume": zero, "vwap": last_close,
            "trade_count": 0, "buy_count": 0, "sell_count": 0,
            "buy_volume": zero, "sell_volume": zero,
            "buy_quote_volume": zero, "sell_quote_volume": zero,
            "trade_intensity": 0.0, "avg_trade_size": zero, "max_trade_size": zero,
            "price_range": zero,
            "tick_count": 0, "up_tick_count": 0, "down_tick_count": 0,
            "volume_imbalance": 0.0,
            "large_trade_count": 0, "large_trade_volume": zero,
        }

    def _compute_orderbook_features(
        self, symbol: str, updates: list[dict]
    ) -> Optional[dict]:
        book = self.current_orderbook.get(symbol, {"bids": {}, "asks": {}})

        sorted_bids = sorted(book["bids"].items(), key=lambda x: x[0], reverse=True)
        sorted_asks = sorted(book["asks"].items(), key=lambda x: x[0])

        if not sorted_bids or not sorted_asks:
            return None

        best_bid_price, best_bid_qty = sorted_bids[0]
        best_ask_price, best_ask_qty = sorted_asks[0]

        spread = best_ask_price - best_bid_price
        mid_price = (best_bid_price + best_ask_price) / 2
        spread_bps = float(spread / mid_price * 10000) if mid_price > 0 else 0.0

        top_5_bids = sorted_bids[:5]
        top_5_asks = sorted_asks[:5]
        bid_depth_5 = sum((qty for _, qty in top_5_bids), Decimal("0"))
        ask_depth_5 = sum((qty for _, qty in top_5_asks), Decimal("0"))
        total_depth_5 = bid_depth_5 + ask_depth_5
        imbalance_5 = (
            float((bid_depth_5 - ask_depth_5) / total_depth_5)
            if total_depth_5 > 0 else 0.0
        )

        bid_thr_02 = mid_price * (Decimal("1") - DEPTH_PCT_02)
        ask_thr_02 = mid_price * (Decimal("1") + DEPTH_PCT_02)
        bid_thr_1 = mid_price * (Decimal("1") - DEPTH_PCT_1)
        ask_thr_1 = mid_price * (Decimal("1") + DEPTH_PCT_1)

        bid_depth_02 = Decimal("0")
        bid_notional_02 = Decimal("0")
        bid_depth_1 = Decimal("0")
        total_bid_depth = Decimal("0")
        for price, qty in sorted_bids:
            total_bid_depth += qty
            if price >= bid_thr_02:
                bid_depth_02 += qty
                bid_notional_02 += price * qty
            if price >= bid_thr_1:
                bid_depth_1 += qty

        ask_depth_02 = Decimal("0")
        ask_notional_02 = Decimal("0")
        ask_depth_1 = Decimal("0")
        total_ask_depth = Decimal("0")
        for price, qty in sorted_asks:
            total_ask_depth += qty
            if price <= ask_thr_02:
                ask_depth_02 += qty
                ask_notional_02 += price * qty
            if price <= ask_thr_1:
                ask_depth_1 += qty

        def _imbalance(bid: Decimal, ask: Decimal) -> float:
            total = bid + ask
            return float((bid - ask) / total) if total > 0 else 0.0

        return {
            "best_bid_price": best_bid_price,
            "best_bid_qty": best_bid_qty,
            "best_ask_price": best_ask_price,
            "best_ask_qty": best_ask_qty,
            "spread_bps": spread_bps,
            "mid_price": mid_price,
            "imbalance_5": imbalance_5,
            "update_count": len(updates),
            "bid_depth_02": float(bid_depth_02),
            "ask_depth_02": float(ask_depth_02),
            "bid_notional_02": float(bid_notional_02),
            "ask_notional_02": float(ask_notional_02),
            "depth_imbalance_02": _imbalance(bid_depth_02, ask_depth_02),
            "bid_depth_1": float(bid_depth_1),
            "ask_depth_1": float(ask_depth_1),
            "depth_imbalance_1": _imbalance(bid_depth_1, ask_depth_1),
            "total_bid_depth": float(total_bid_depth),
            "total_ask_depth": float(total_ask_depth),
            "depth_imbalance_total": _imbalance(total_bid_depth, total_ask_depth),
        }

    @staticmethod
    def _compute_trade_features(trades: list[dict]) -> Optional[dict]:
        """计算分钟内的成交特征（含高频特征）。"""
        if not trades:
            return None

        df = pd.DataFrame(trades)

        open_price = df.iloc[0]["price"]
        close_price = df.iloc[-1]["price"]
        high_price = df["price"].max()
        low_price = df["price"].min()

        volume = df["quantity"].sum()
        quote_volume = (df["price"] * df["quantity"]).sum()
        trade_count = len(df)

        buy_trades = df[~df["is_buyer_maker"]]
        sell_trades = df[df["is_buyer_maker"]]

        buy_volume = buy_trades["quantity"].sum() if len(buy_trades) > 0 else Decimal("0")
        sell_volume = sell_trades["quantity"].sum() if len(sell_trades) > 0 else Decimal("0")
        buy_quote_volume = (
            (buy_trades["price"] * buy_trades["quantity"]).sum() if len(buy_trades) > 0 else Decimal("0")
        )
        sell_quote_volume = (
            (sell_trades["price"] * sell_trades["quantity"]).sum() if len(sell_trades) > 0 else Decimal("0")
        )
        buy_count = len(buy_trades)
        sell_count = len(sell_trades)

        trade_intensity = float(trade_count)
        avg_trade_size = volume / trade_count if trade_count > 0 else Decimal("0")
        max_trade_size = df["quantity"].max()
        price_range = high_price - low_price

        prices = df["price"].tolist()
        price_diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        tick_count = sum(1 for d in price_diffs if d != 0)
        up_tick_count = sum(1 for d in price_diffs if d > 0)
        down_tick_count = sum(1 for d in price_diffs if d < 0)

        total_vol = float(volume) if volume else 0.0
        vol_imbalance = (
            (float(buy_volume) - float(sell_volume)) / total_vol if total_vol > 0 else 0.0
        )

        quantities = df["quantity"].tolist()
        if len(quantities) > 1:
            mean_q = volume / trade_count
            std_q = (
                sum((q - mean_q) ** 2 for q in quantities) / len(quantities)
            ) ** Decimal("0.5")
            threshold = mean_q + 2 * std_q
            large_trades = [q for q in quantities if q > threshold]
            large_trade_count = len(large_trades)
            large_trade_volume = sum(large_trades, Decimal("0"))
        else:
            large_trade_count = 0
            large_trade_volume = Decimal("0")

        vwap_val: Optional[Decimal] = (quote_volume / volume) if volume > 0 else None

        return {
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "quote_volume": quote_volume,
            "vwap": vwap_val,
            "trade_count": trade_count,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "buy_quote_volume": buy_quote_volume,
            "sell_quote_volume": sell_quote_volume,
            "trade_intensity": trade_intensity,
            "avg_trade_size": avg_trade_size,
            "max_trade_size": max_trade_size,
            "price_range": price_range,
            "tick_count": tick_count,
            "up_tick_count": up_tick_count,
            "down_tick_count": down_tick_count,
            "volume_imbalance": vol_imbalance,
            "large_trade_count": large_trade_count,
            "large_trade_volume": large_trade_volume,
        }
