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

        # 事件时间水位线（最大已见事件秒数）
        # 聚合基于 WebSocket 事件时间而非本地时钟
        self.max_event_second = 0
        # 已刷新的秒数（避免重复）
        self.last_flushed_second = 0
        # 水位线延迟：等待该秒不再有新事件到达
        self.watermark_delay_seconds = 1

        # 聚合任务
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

        使用 WebSocket 事件时间（而非本地时钟）驱动聚合：
        - 当水位线前进到 T，说明时间 <= T - watermark_delay 的秒已不再可能收到新事件
        - 刷新所有 last_flushed_second < s <= flushable_second 的秒
        """
        while self.running:
            try:
                await asyncio.sleep(0.1)  # 高频检查水位线

                if self.max_event_second == 0:
                    continue

                # 首次收到事件时初始化刷新游标
                if self.last_flushed_second == 0:
                    self.last_flushed_second = self.max_event_second - self.watermark_delay_seconds - 1

                # 可安全刷新的最大秒数
                flushable_second = self.max_event_second - self.watermark_delay_seconds

                # 从上次刷新点到当前可刷点，逐秒聚合
                for second in range(self.last_flushed_second + 1, flushable_second + 1):
                    # 该秒有数据才聚合，没数据也要推进水位线
                    if second in self.orderbook_buffer or second in self.trade_buffer:
                        await self._aggregate_second(second)
                        self.orderbook_buffer.pop(second, None)
                        self.trade_buffer.pop(second, None)
                    self.last_flushed_second = second

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"聚合循环错误: {e}", exc_info=True)

    async def _aggregate_second(self, timestamp_second: int):
        """聚合某一秒的数据（合并订单簿和成交）"""
        # 按交易对收集订单簿更新
        ob_by_symbol = defaultdict(list)
        for item in self.orderbook_buffer.get(timestamp_second, []):
            ob_by_symbol[item['symbol']].append(item['data'])

        # 按交易对收集成交
        tr_by_symbol = defaultdict(list)
        for item in self.trade_buffer.get(timestamp_second, []):
            tr_by_symbol[item['symbol']].append(item['data'])

        # 合并出现的所有交易对
        all_symbols = set(ob_by_symbol.keys()) | set(tr_by_symbol.keys())

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

                unified = UnifiedMarketData(
                    timestamp=datetime.fromtimestamp(timestamp_second, tz=timezone.utc),
                    symbol=symbol,
                    # 订单簿特征
                    best_bid_price=ob_feat['best_bid_price'] if ob_feat else None,
                    best_bid_qty=ob_feat['best_bid_qty'] if ob_feat else None,
                    best_ask_price=ob_feat['best_ask_price'] if ob_feat else None,
                    best_ask_qty=ob_feat['best_ask_qty'] if ob_feat else None,
                    spread_bps=ob_feat['spread_bps'] if ob_feat else None,
                    mid_price=ob_feat['mid_price'] if ob_feat else None,
                    imbalance_5=ob_feat['imbalance_5'] if ob_feat else None,
                    update_count=ob_feat['update_count'] if ob_feat else 0,
                    # 成交特征
                    open=tr_feat['open'] if tr_feat else None,
                    high=tr_feat['high'] if tr_feat else None,
                    low=tr_feat['low'] if tr_feat else None,
                    close=tr_feat['close'] if tr_feat else None,
                    volume=tr_feat['volume'] if tr_feat else None,
                    vwap=tr_feat['vwap'] if tr_feat else None,
                    trade_count=tr_feat['trade_count'] if tr_feat else 0,
                    buy_count=tr_feat['buy_count'] if tr_feat else 0,
                    sell_count=tr_feat['sell_count'] if tr_feat else 0,
                    buy_volume=tr_feat['buy_volume'] if tr_feat else None,
                    sell_volume=tr_feat['sell_volume'] if tr_feat else None,
                )

                if self.on_aggregated_data:
                    self.on_aggregated_data(unified)

            except Exception as e:
                logger.error(f"数据聚合错误 {symbol}: {e}", exc_info=True)

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
        """计算成交特征，返回字典或 None（当成交为空）"""
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

        vwap = quote_volume / volume if volume > 0 else Decimal('0')

        buy_trades = df[~df['is_buyer_maker']]
        sell_trades = df[df['is_buyer_maker']]

        buy_volume = buy_trades['quantity'].sum() if len(buy_trades) > 0 else Decimal('0')
        sell_volume = sell_trades['quantity'].sum() if len(sell_trades) > 0 else Decimal('0')
        buy_count = len(buy_trades)
        sell_count = len(sell_trades)

        return {
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'vwap': vwap,
            'trade_count': trade_count,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
        }
