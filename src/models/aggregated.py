"""聚合数据模型（按秒聚合）"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional


@dataclass(frozen=True)
class AggregatedOrderBook:
    """秒级聚合订单簿数据"""
    timestamp: datetime  # 聚合时间戳（秒级）
    exchange: str
    symbol: str

    # 最优买卖价
    best_bid_price: Decimal
    best_bid_qty: Decimal
    best_ask_price: Decimal
    best_ask_qty: Decimal

    # 价差
    spread: Decimal
    spread_bps: float  # 基点
    mid_price: Decimal

    # 深度统计（前5档）
    bid_depth_5: Decimal
    ask_depth_5: Decimal
    bid_volume_5: Decimal
    ask_volume_5: Decimal

    # 不平衡度
    imbalance_5: float  # (bid - ask) / (bid + ask)

    # 更新统计
    update_count: int  # 该秒内的更新次数


@dataclass(frozen=True)
class AggregatedTrades:
    """秒级聚合成交数据"""
    timestamp: datetime  # 聚合时间戳（秒级）
    exchange: str
    symbol: str

    # 价格统计
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    vwap: Decimal  # 成交量加权平均价

    # 成交量统计
    volume: Decimal
    quote_volume: Decimal
    trade_count: int

    # 买卖方向统计
    buy_volume: Decimal
    sell_volume: Decimal
    buy_count: int
    sell_count: int

    # 大单统计（可选）
    large_trade_count: int
    large_trade_volume: Decimal


@dataclass(frozen=True)
class UnifiedMarketData:
    """统一市场数据模型（按秒聚合）

    将订单簿特征、成交特征和高频交易特征合并到单一结构中。
    历史模式下订单簿字段为 None，实时模式下全部可用。
    """
    # 索引字段
    timestamp: datetime
    symbol: str

    # OHLCV
    open: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    close: Optional[Decimal] = None
    volume: Optional[Decimal] = None
    quote_volume: Optional[Decimal] = None
    vwap: Optional[Decimal] = None

    # 成交统计
    trade_count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    buy_volume: Optional[Decimal] = None
    sell_volume: Optional[Decimal] = None
    buy_quote_volume: Optional[Decimal] = None
    sell_quote_volume: Optional[Decimal] = None

    # 高频特征
    trade_intensity: float = 0.0
    avg_trade_size: Optional[Decimal] = None
    max_trade_size: Optional[Decimal] = None
    price_range: Optional[Decimal] = None
    tick_count: int = 0
    up_tick_count: int = 0
    down_tick_count: int = 0
    volume_imbalance: float = 0.0
    large_trade_count: int = 0
    large_trade_volume: Optional[Decimal] = None

    # 订单簿特征（实时 depth@100ms 流计算）
    best_bid_price: Optional[Decimal] = None
    best_bid_qty: Optional[Decimal] = None
    best_ask_price: Optional[Decimal] = None
    best_ask_qty: Optional[Decimal] = None
    spread_bps: Optional[float] = None
    mid_price: Optional[Decimal] = None
    imbalance_5: Optional[float] = None
    update_count: int = 0

    # 订单簿深度特征（历史 bookDepth 数据）
    bid_depth_02: Optional[float] = None
    ask_depth_02: Optional[float] = None
    bid_notional_02: Optional[float] = None
    ask_notional_02: Optional[float] = None
    depth_imbalance_02: Optional[float] = None
    bid_depth_1: Optional[float] = None
    ask_depth_1: Optional[float] = None
    depth_imbalance_1: Optional[float] = None
    total_bid_depth: Optional[float] = None
    total_ask_depth: Optional[float] = None
    depth_imbalance_total: Optional[float] = None

    # K线特征（分钟级，forward fill 到秒）
    kline_open: Optional[float] = None
    kline_high: Optional[float] = None
    kline_low: Optional[float] = None
    kline_close: Optional[float] = None
    kline_volume: Optional[float] = None
    kline_quote_volume: Optional[float] = None
    kline_count: Optional[int] = None
    kline_taker_buy_volume: Optional[float] = None
    kline_taker_buy_quote_volume: Optional[float] = None
    kline_taker_buy_ratio: Optional[float] = None
    kline_body_ratio: Optional[float] = None
    kline_upper_shadow: Optional[float] = None
    kline_lower_shadow: Optional[float] = None
