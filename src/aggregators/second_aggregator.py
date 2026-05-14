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


# 与 history_collector 的 bookDepth 档位口径保持一致：
# percentage 负值代表 bid 侧偏离 mid 的百分比，正值代表 ask 侧
DEPTH_PCT_02 = Decimal("0.002")  # ±0.2%
DEPTH_PCT_1 = Decimal("0.01")    # ±1%


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
        # K线流：每 symbol 仅保留最新快照（含进行中的当前分钟），按秒 ffill 用
        self.last_kline: dict[str, dict] = {}

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

    def add_kline(self, symbol: str, data: dict):
        """添加 K 线快照（kline_<interval> 流）

        K 线流每 1~2 秒推送一次进行中（未关闭）的当前 K 线，K 线关闭时再推一次。
        我们按 symbol 仅保留最新快照，由聚合循环按秒 ffill 到 UnifiedMarketData。

        Args:
            symbol: 交易对
            data: 由 BinanceCollector.parse_kline 解析后的 dict
        """
        self.last_kline[symbol] = data

        # 将 event_time 也纳入水位线，避免在仅有 kline 流时阻塞输出
        timestamp_ms = data.get('event_time') or data.get('close_time')
        if timestamp_ms:
            timestamp_second = self._get_second_timestamp(timestamp_ms)
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
                kline_feat = self._compute_kline_features(self.last_kline.get(symbol))

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
                    # 实时 depth@100ms 衍生（top-of-book + 5 档）
                    best_bid_price=ob_feat['best_bid_price'] if ob_feat else None,
                    best_bid_qty=ob_feat['best_bid_qty'] if ob_feat else None,
                    best_ask_price=ob_feat['best_ask_price'] if ob_feat else None,
                    best_ask_qty=ob_feat['best_ask_qty'] if ob_feat else None,
                    spread_bps=ob_feat['spread_bps'] if ob_feat else None,
                    mid_price=ob_feat['mid_price'] if ob_feat else None,
                    imbalance_5=ob_feat['imbalance_5'] if ob_feat else None,
                    update_count=ob_feat['update_count'] if ob_feat else 0,
                    # 订单簿深度（与 history bookDepth 字段口径对齐）
                    bid_depth_02=ob_feat['bid_depth_02'] if ob_feat else None,
                    ask_depth_02=ob_feat['ask_depth_02'] if ob_feat else None,
                    bid_notional_02=ob_feat['bid_notional_02'] if ob_feat else None,
                    ask_notional_02=ob_feat['ask_notional_02'] if ob_feat else None,
                    depth_imbalance_02=ob_feat['depth_imbalance_02'] if ob_feat else None,
                    bid_depth_1=ob_feat['bid_depth_1'] if ob_feat else None,
                    ask_depth_1=ob_feat['ask_depth_1'] if ob_feat else None,
                    depth_imbalance_1=ob_feat['depth_imbalance_1'] if ob_feat else None,
                    total_bid_depth=ob_feat['total_bid_depth'] if ob_feat else None,
                    total_ask_depth=ob_feat['total_ask_depth'] if ob_feat else None,
                    depth_imbalance_total=ob_feat['depth_imbalance_total'] if ob_feat else None,
                    # K 线特征（来自 kline_<interval> 流，按秒 ffill）
                    kline_open=kline_feat['kline_open'] if kline_feat else None,
                    kline_high=kline_feat['kline_high'] if kline_feat else None,
                    kline_low=kline_feat['kline_low'] if kline_feat else None,
                    kline_close=kline_feat['kline_close'] if kline_feat else None,
                    kline_volume=kline_feat['kline_volume'] if kline_feat else None,
                    kline_quote_volume=kline_feat['kline_quote_volume'] if kline_feat else None,
                    kline_count=kline_feat['kline_count'] if kline_feat else None,
                    kline_taker_buy_volume=kline_feat['kline_taker_buy_volume'] if kline_feat else None,
                    kline_taker_buy_quote_volume=kline_feat['kline_taker_buy_quote_volume'] if kline_feat else None,
                    kline_taker_buy_ratio=kline_feat['kline_taker_buy_ratio'] if kline_feat else None,
                    kline_body_ratio=kline_feat['kline_body_ratio'] if kline_feat else None,
                    kline_upper_shadow=kline_feat['kline_upper_shadow'] if kline_feat else None,
                    kline_lower_shadow=kline_feat['kline_lower_shadow'] if kline_feat else None,
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

        total_depth_5 = bid_depth_5 + ask_depth_5
        imbalance_5 = (
            float((bid_depth_5 - ask_depth_5) / total_depth_5)
            if total_depth_5 > 0 else 0.0
        )

        # 与 history bookDepth 对齐：±0.2% / ±1% / 全簿 累计深度与名义额
        bid_thr_02 = mid_price * (Decimal('1') - DEPTH_PCT_02)
        ask_thr_02 = mid_price * (Decimal('1') + DEPTH_PCT_02)
        bid_thr_1 = mid_price * (Decimal('1') - DEPTH_PCT_1)
        ask_thr_1 = mid_price * (Decimal('1') + DEPTH_PCT_1)

        bid_depth_02 = Decimal('0')
        bid_notional_02 = Decimal('0')
        bid_depth_1 = Decimal('0')
        total_bid_depth = Decimal('0')
        for price, qty in sorted_bids:
            total_bid_depth += qty
            if price >= bid_thr_02:
                bid_depth_02 += qty
                bid_notional_02 += price * qty
            if price >= bid_thr_1:
                bid_depth_1 += qty

        ask_depth_02 = Decimal('0')
        ask_notional_02 = Decimal('0')
        ask_depth_1 = Decimal('0')
        total_ask_depth = Decimal('0')
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
            'best_bid_price': best_bid_price,
            'best_bid_qty': best_bid_qty,
            'best_ask_price': best_ask_price,
            'best_ask_qty': best_ask_qty,
            'spread_bps': spread_bps,
            'mid_price': mid_price,
            'imbalance_5': imbalance_5,
            'update_count': len(updates),
            'bid_depth_02': float(bid_depth_02),
            'ask_depth_02': float(ask_depth_02),
            'bid_notional_02': float(bid_notional_02),
            'ask_notional_02': float(ask_notional_02),
            'depth_imbalance_02': _imbalance(bid_depth_02, ask_depth_02),
            'bid_depth_1': float(bid_depth_1),
            'ask_depth_1': float(ask_depth_1),
            'depth_imbalance_1': _imbalance(bid_depth_1, ask_depth_1),
            'total_bid_depth': float(total_bid_depth),
            'total_ask_depth': float(total_ask_depth),
            'depth_imbalance_total': _imbalance(total_bid_depth, total_ask_depth),
        }

    @staticmethod
    def _compute_kline_features(kline: Optional[dict]) -> Optional[dict]:
        """从最新 K 线快照计算 UnifiedMarketData 中的 kline_* 特征。

        与 history_collector._compute_kline_features 保持口径一致：
          - kline_taker_buy_ratio: taker_buy_volume / volume（volume==0 → 0.5）
          - kline_body_ratio:      |close-open| / (high-low)（区间为 0 → 0）
          - kline_upper_shadow:    (high - max(open,close)) / (high-low)
          - kline_lower_shadow:    (min(open,close) - low) / (high-low)
        """
        if kline is None:
            return None

        open_p = kline['open']
        high_p = kline['high']
        low_p = kline['low']
        close_p = kline['close']
        volume = kline['volume']
        taker_buy_volume = kline['taker_buy_volume']

        hl_range = high_p - low_p
        body = abs(close_p - open_p)
        upper_shadow = high_p - max(open_p, close_p)
        lower_shadow = min(open_p, close_p) - low_p

        taker_buy_ratio = (
            float(taker_buy_volume / volume) if volume > 0 else 0.5
        )
        body_ratio = float(body / hl_range) if hl_range > 0 else 0.0
        upper_ratio = float(upper_shadow / hl_range) if hl_range > 0 else 0.0
        lower_ratio = float(lower_shadow / hl_range) if hl_range > 0 else 0.0

        return {
            'kline_open': float(open_p),
            'kline_high': float(high_p),
            'kline_low': float(low_p),
            'kline_close': float(close_p),
            'kline_volume': float(volume),
            'kline_quote_volume': float(kline['quote_volume']),
            'kline_count': int(kline['count']),
            'kline_taker_buy_volume': float(taker_buy_volume),
            'kline_taker_buy_quote_volume': float(kline['taker_buy_quote_volume']),
            'kline_taker_buy_ratio': taker_buy_ratio,
            'kline_body_ratio': body_ratio,
            'kline_upper_shadow': upper_ratio,
            'kline_lower_shadow': lower_ratio,
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
