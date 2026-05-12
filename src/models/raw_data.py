"""原始数据模型"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True)
class OrderBookUpdate:
    """订单簿增量更新（depth@100ms）"""
    timestamp: datetime
    exchange: str
    symbol: str
    event_time: int  # 事件时间（毫秒）
    first_update_id: int
    final_update_id: int
    bids: list[tuple[Decimal, Decimal]]  # [(price, quantity), ...]
    asks: list[tuple[Decimal, Decimal]]


@dataclass(frozen=True)
class Trade:
    """成交记录"""
    timestamp: datetime
    exchange: str
    symbol: str
    trade_id: int
    price: Decimal
    quantity: Decimal
    buyer_order_id: int
    seller_order_id: int
    trade_time: int  # 成交时间（毫秒）
    is_buyer_maker: bool  # 买方是否为 maker


@dataclass(frozen=True)
class BookTicker:
    """最优买卖价（bookTicker）"""
    timestamp: datetime
    exchange: str
    symbol: str
    update_id: int
    best_bid_price: Decimal
    best_bid_qty: Decimal
    best_ask_price: Decimal
    best_ask_qty: Decimal


@dataclass(frozen=True)
class DerivativeTicker:
    """衍生品行情（24hr ticker）"""
    timestamp: datetime
    exchange: str
    symbol: str
    price_change: Decimal
    price_change_percent: Decimal
    weighted_avg_price: Decimal
    last_price: Decimal
    last_qty: Decimal
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    volume: Decimal
    quote_volume: Decimal
    open_time: int
    close_time: int
    first_trade_id: int
    last_trade_id: int
    trade_count: int


@dataclass(frozen=True)
class Liquidation:
    """强平数据（forceOrder）"""
    timestamp: datetime
    exchange: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    order_type: str  # 'LIMIT' or 'MARKET'
    time_in_force: str
    original_quantity: Decimal
    price: Decimal
    avg_price: Decimal
    order_status: str
    order_last_filled_qty: Decimal
    order_filled_accumulated_qty: Decimal
    order_trade_time: int
