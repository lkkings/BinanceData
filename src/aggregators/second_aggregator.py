"""秒级时间戳聚合器"""
import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable, Optional

import pandas as pd

from ..models.aggregated import UnifiedMarketData

logger = logging.getLogger(__name__)


class SecondAggregator:
    """秒级数据聚合器

    将实时数据按秒为单位进行聚合，生成秒级特征数据。
    订单簿和成交会被合并到单一的 UnifiedMarketData 中返回。
    """

    def __init__(
        self,
        on_aggregated_data: Callable[[UnifiedMarketData], None] | None = None
    ):
        """初始化聚合器

        Args:
            on_aggregated_data: 聚合完成后的统一数据回调
        """
        self.on_aggregated_data = on_aggregated_data

        # 数据缓冲区（按秒分组）
        self.orderbook_buffer = defaultdict(list)  # {timestamp_second: [updates]}
        self.trade_buffer = defaultdict(list)      # {timestamp_second: [trades]}

        # 当前订单簿状态（用于计算深度）
        self.current_orderbook = {}  # {symbol: {'bids': {}, 'asks': {}}}

        # 上一秒的收盘价（用于零成交秒 OHLC 填充，按 symbol 分）
        self.last_close: dict[str, Decimal] = {}

        # 事件时间水位线（最大已见事件秒数）
        self.max_event_second = 0
        self.last_flushed_second = 0
        self.watermark_delay_seconds = 1

        self.aggregation_task = None
        self.running = False

    async def start(self):
        """启动聚合任务"""
        if not self.running:
            self.running = True
            self.aggregation_task = asyncio.create_task(self._aggregation_loop())
            logger.info("秒级聚合器已启动")

    async def stop(self):
        """停止聚合任务"""
        self.running = False
        if self.aggregation_task:
            self.aggregation_task.cancel()
            try:
                await self.aggregation_task
            except asyncio.CancelledError:
                pass
        logger.info("秒级聚合器已停止")

    def add_orderbook_update(self, symbol: str, data: dict):
        """添加订单簿更新

        Args:
            symbol: 交易对
            data: 订单簿数据
        """
        timestamp_ms = data['event_time']
        timestamp_second = self._get_second_timestamp(timestamp_ms)

        self.orderbook_buffer[timestamp_second].append({
            'symbol': symbol,
            'data': data
        })

        self._update_orderbook_state(symbol, data)

        # 更新事件时间水位线
        if timestamp_second > self.max_event_second:
            self.max_event_second = timestamp_second

    def add_trade(self, symbol: str, data: dict):
        """添加成交记录

        Args:
            symbol: 交易对
            data: 成交数据
        """
        timestamp_ms = data['trade_time']
        timestamp_second = self._get_second_timestamp(timestamp_ms)

        self.trade_buffer[timestamp_second].append({
            'symbol': symbol,
            'data': data
        })

        # 更新事件时间水位线
        if timestamp_second > self.max_event_second:
            self.max_event_second = timestamp_second

    def _get_second_timestamp(self, timestamp_ms: int) -> int:
        """将毫秒时间戳转换为秒级时间戳"""
        return timestamp_ms // 1000

    def _update_orderbook_state(self, symbol: str, data: dict):
        """更新订单簿状态"""
        if symbol not in self.current_orderbook:
            self.current_orderbook[symbol] = {
                'bids': {},
                'asks': {}
            }

        book = self.current_orderbook[symbol]

        for price, qty in data['bids']:
            if qty == 0:
                book['bids'].pop(price, None)
            else:
                book['bids'][price] = qty

        for price, qty in data['asks']:
            if qty == 0:
                book['asks'].pop(price, None)
            else:
                book['asks'][price] = qty

    async def _aggregation_loop(self):
        """聚合循环（基于事件时间水位线）

        每秒严格产出一条记录（含零成交秒），保证下游拿到连续时间索引。
        """
        while self.running:
            try:
                await asyncio.sleep(0.1)

                if self.max_event_second == 0:
                    continue

                if self.last_flushed_second == 0:
                    self.last_flushed_second = self.max_event_second - self.watermark_delay_seconds - 1

                flushable_second = self.max_event_second - self.watermark_delay_seconds

                for second in range(self.last_flushed_second + 1, flushable_second + 1):
                    await self._aggregate_second(second)
                    self.orderbook_buffer.pop(second, None)
                    self.trade_buffer.pop(second, None)
                    self.last_flushed_second = second

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"聚合循环错误: {e}", exc_info=True)

    async def _aggregate_second(self, timestamp_second: int):
        """聚合某一秒的数据（订单簿 + 成交）。零成交秒也产出记录。"""
        ob_by_symbol = defaultdict(list)
        for item in self.orderbook_buffer.get(timestamp_second, []):
            ob_by_symbol[item['symbol']].append(item['data'])

        tr_by_symbol = defaultdict(list)
        for item in self.trade_buffer.get(timestamp_second, []):
            tr_by_symbol[item['symbol']].append(item['data'])

        # 即使当前秒没有新事件，也对已知 symbol 输出一条（用上次 close 填充）
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

                # 零成交秒：OHLC 用上一秒 close 填充，量类为 0
                if tr_feat is None:
                    last = self.last_close.get(symbol)
                    if last is None:
                        # 尚未有过任何成交，跳过本 symbol
                        continue
                    tr_feat = self._empty_trade_features(last)
                else:
                    self.last_close[symbol] = tr_feat['close']

                unified = UnifiedMarketData(
                    timestamp=datetime.fromtimestamp(timestamp_second, tz=timezone.utc),
                    symbol=symbol,
                    open=tr_feat['open'],
                    high=tr_feat['high'],
                    low=tr_feat['low'],
                    close=tr_feat['close'],
                    volume=tr_feat['volume'],
                    quote_volume=tr_feat['quote_volume'],
                    vwap=tr_feat['vwap'],
                    trade_count=tr_feat['trade_count'],
                    buy_count=tr_feat['buy_count'],
                    sell_count=tr_feat['sell_count'],
                    buy_volume=tr_feat['buy_volume'],
                    sell_volume=tr_feat['sell_volume'],
                    buy_quote_volume=tr_feat['buy_quote_volume'],
                    sell_quote_volume=tr_feat['sell_quote_volume'],
                    trade_intensity=tr_feat['trade_intensity'],
                    avg_trade_size=tr_feat['avg_trade_size'],
                    max_trade_size=tr_feat['max_trade_size'],
                    price_range=tr_feat['price_range'],
                    tick_count=tr_feat['tick_count'],
                    up_tick_count=tr_feat['up_tick_count'],
                    down_tick_count=tr_feat['down_tick_count'],
                    volume_imbalance=tr_feat['volume_imbalance'],
                    large_trade_count=tr_feat['large_trade_count'],
                    large_trade_volume=tr_feat['large_trade_volume'],
                    best_bid_price=ob_feat['best_bid_price'] if ob_feat else None,
                    best_bid_qty=ob_feat['best_bid_qty'] if ob_feat else None,
                    best_ask_price=ob_feat['best_ask_price'] if ob_feat else None,
                    best_ask_qty=ob_feat['best_ask_qty'] if ob_feat else None,
                    spread_bps=ob_feat['spread_bps'] if ob_feat else None,
                    mid_price=ob_feat['mid_price'] if ob_feat else None,
                    imbalance_5=ob_feat['imbalance_5'] if ob_feat else None,
                    update_count=ob_feat['update_count'] if ob_feat else 0,
                )

                if self.on_aggregated_data:
                    self.on_aggregated_data(unified)

            except Exception as e:
                logger.error(f"数据聚合错误 {symbol}: {e}", exc_info=True)

    @staticmethod
    def _empty_trade_features(last_close: Decimal) -> dict:
        """零成交秒的成交特征：OHLC=last_close，量/计数为 0"""
        zero = Decimal('0')
        return {
            'open': last_close, 'high': last_close, 'low': last_close, 'close': last_close,
            'volume': zero, 'quote_volume': zero, 'vwap': last_close,
            'trade_count': 0, 'buy_count': 0, 'sell_count': 0,
            'buy_volume': zero, 'sell_volume': zero,
            'buy_quote_volume': zero, 'sell_quote_volume': zero,
            'trade_intensity': 0.0, 'avg_trade_size': zero, 'max_trade_size': zero,
            'price_range': zero,
            'tick_count': 0, 'up_tick_count': 0, 'down_tick_count': 0,
            'volume_imbalance': 0.0,
            'large_trade_count': 0, 'large_trade_volume': zero,
        }

    def _compute_orderbook_features(
        self,
        symbol: str,
        updates: list[dict]
    ) -> Optional[dict]:
        """计算订单簿特征，返回字典或 None（当订单簿为空）"""
        book = self.current_orderbook.get(symbol, {'bids': {}, 'asks': {}})

        sorted_bids = sorted(book['bids'].items(), key=lambda x: x[0], reverse=True)
        sorted_asks = sorted(book['asks'].items(), key=lambda x: x[0])

        if not sorted_bids or not sorted_asks:
            return None

        best_bid_price, best_bid_qty = sorted_bids[0]
        best_ask_price, best_ask_qty = sorted_asks[0]

        spread = best_ask_price - best_bid_price
        mid_price = (best_bid_price + best_ask_price) / 2
        spread_bps = float(spread / mid_price * 10000) if mid_price > 0 else 0.0

        top_5_bids = sorted_bids[:5]
        top_5_asks = sorted_asks[:5]

        bid_depth_5 = sum((qty for _, qty in top_5_bids), Decimal('0'))
        ask_depth_5 = sum((qty for _, qty in top_5_asks), Decimal('0'))

        total_depth = bid_depth_5 + ask_depth_5
        imbalance_5 = (
            float((bid_depth_5 - ask_depth_5) / total_depth)
            if total_depth > 0 else 0.0
        )

        return {
            'best_bid_price': best_bid_price,
            'best_bid_qty': best_bid_qty,
            'best_ask_price': best_ask_price,
            'best_ask_qty': best_ask_qty,
            'spread_bps': spread_bps,
            'mid_price': mid_price,
            'imbalance_5': imbalance_5,
            'update_count': len(updates),
        }

    def _compute_trade_features(self, trades: list[dict]) -> Optional[dict]:
        """计算成交特征（含高频特征），返回字典或 None（当成交为空）"""
        if not trades:
            return None

        df = pd.DataFrame(trades)

        open_price = df.iloc[0]['price']
        close_price = df.iloc[-1]['price']
        high_price = df['price'].max()
        low_price = df['price'].min()

        volume = df['quantity'].sum()
        quote_volume = (df['price'] * df['quantity']).sum()
        trade_count = len(df)

        buy_trades = df[~df['is_buyer_maker']]
        sell_trades = df[df['is_buyer_maker']]

        buy_volume = buy_trades['quantity'].sum() if len(buy_trades) > 0 else Decimal('0')
        sell_volume = sell_trades['quantity'].sum() if len(sell_trades) > 0 else Decimal('0')
        buy_quote_volume = (buy_trades['price'] * buy_trades['quantity']).sum() if len(buy_trades) > 0 else Decimal('0')
        sell_quote_volume = (sell_trades['price'] * sell_trades['quantity']).sum() if len(sell_trades) > 0 else Decimal('0')
        buy_count = len(buy_trades)
        sell_count = len(sell_trades)

        # 高频特征
        trade_intensity = float(trade_count)
        avg_trade_size = volume / trade_count if trade_count > 0 else Decimal('0')
        max_trade_size = df['quantity'].max()
        price_range = high_price - low_price

        prices = df['price'].tolist()
        price_diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        tick_count = sum(1 for d in price_diffs if d != 0)
        up_tick_count = sum(1 for d in price_diffs if d > 0)
        down_tick_count = sum(1 for d in price_diffs if d < 0)

        total_vol = float(volume) if volume else 0.0
        vol_imbalance = (float(buy_volume) - float(sell_volume)) / total_vol if total_vol > 0 else 0.0

        # 大单检测
        quantities = df['quantity'].tolist()
        if len(quantities) > 1:
            mean_q = volume / trade_count
            std_q = (sum((q - mean_q) ** 2 for q in quantities) / len(quantities)) ** Decimal('0.5')
            threshold = mean_q + 2 * std_q
            large_trades = [q for q in quantities if q > threshold]
            large_trade_count = len(large_trades)
            large_trade_volume = sum(large_trades, Decimal('0'))
        else:
            large_trade_count = 0
            large_trade_volume = Decimal('0')

        # vwap：volume==0 时无定义，返回 None 让下游识别"无成交"语义
        vwap_val: Optional[Decimal] = (quote_volume / volume) if volume > 0 else None

        return {
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'quote_volume': quote_volume,
            'vwap': vwap_val,
            'trade_count': trade_count,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'buy_quote_volume': buy_quote_volume,
            'sell_quote_volume': sell_quote_volume,
            'trade_intensity': trade_intensity,
            'avg_trade_size': avg_trade_size,
            'max_trade_size': max_trade_size,
            'price_range': price_range,
            'tick_count': tick_count,
            'up_tick_count': up_tick_count,
            'down_tick_count': down_tick_count,
            'volume_imbalance': vol_imbalance,
            'large_trade_count': large_trade_count,
            'large_trade_volume': large_trade_volume,
        }
