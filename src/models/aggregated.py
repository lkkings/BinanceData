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

    将订单簿特征和成交特征合并到单一结构中。
    订单簿或成交缺失时，对应字段为 None。
    """
    # 索引字段
    timestamp: datetime
    symbol: str

    # 订单簿特征
    best_bid_price: Optional[Decimal] = None
    best_bid_qty: Optional[Decimal] = None
    best_ask_price: Optional[Decimal] = None
    best_ask_qty: Optional[Decimal] = None
    spread_bps: Optional[float] = None
    mid_price: Optional[Decimal] = None
    imbalance_5: Optional[float] = None
    update_count: int = 0

    # 成交特征
    open: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    close: Optional[Decimal] = None
    volume: Optional[Decimal] = None
    vwap: Optional[Decimal] = None
    trade_count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    buy_volume: Optional[Decimal] = None
    sell_volume: Optional[Decimal] = None
