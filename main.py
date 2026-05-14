"""主程序入口 - 优先历史数据下载，回退 WebSocket 实时采集"""
import asyncio
import logging
from dataclasses import fields
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.collectors import BinanceCollector, HistoryCollector
from src.aggregators import SecondAggregator
from src.config import get_settings
from src.models import UnifiedMarketData

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _to_float(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return value


def unified_to_row(data: UnifiedMarketData) -> dict:
    """将 UnifiedMarketData 转为 dict，Decimal -> float"""
    row = {}
    for f in fields(data):
        val = getattr(data, f.name)
        if f.name == 'timestamp':
            row[f.name] = val.strftime('%Y-%m-%d %H:%M:%S')
        else:
            row[f.name] = _to_float(val)
    return row


def save_aggregated_data(records: list[UnifiedMarketData], symbol: str, date_str: str):
    """保存聚合数据到 CSV，按秒级时间戳严格连续，缺失秒自动补齐"""
    settings = get_settings()
    if not records:
        return

    rows = [unified_to_row(r) for r in records]
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    # 对齐到严格连续的秒级索引
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1s')
    missing_count = len(full_range) - len(df)
    df = df.reindex(full_range)
    df.index.name = 'timestamp'

    df['symbol'] = symbol

    # 缺失秒的 OHLC/vwap 退化为上一秒 close（无成交 → 价格横线）
    missing_mask = df['close'].isna()
    if missing_mask.any():
        close_ref = df['close'].ffill().bfill()
        for col in ('open', 'high', 'low', 'close', 'vwap'):
            if col in df.columns:
                df.loc[missing_mask, col] = close_ref[missing_mask]

    # K 线字段按分钟对齐，ffill + bfill（开头那分钟是已知的，不构成未来函数）
    kline_cols = [c for c in df.columns if c.startswith('kline_')]
    for col in kline_cols:
        df[col] = df[col].ffill().bfill()

    # 订单簿/深度字段只 ffill（避免 bfill 把未来快照泄露到数据集开头）
    depth_cols = [c for c in df.columns if c.startswith((
        'bid_depth', 'ask_depth', 'bid_notional', 'ask_notional', 'depth_imbalance',
        'total_bid', 'total_ask',
        'best_bid', 'best_ask', 'spread_bps', 'mid_price', 'imbalance_5',
    ))]
    depth_age: Optional[pd.Series] = None
    if depth_cols:
        # 以首个可用列作为快照"是否刷新"的代表
        ref_col = next((c for c in ('bid_depth_02', 'total_bid_depth', 'best_bid_price')
                        if c in df.columns), depth_cols[0])
        has_snapshot = df[ref_col].notna()
        snapshot_idx = pd.Series(np.where(has_snapshot, np.arange(len(df)), np.nan), index=df.index)
        last_snapshot_idx = snapshot_idx.ffill()
        depth_age = (np.arange(len(df)) - last_snapshot_idx)  # NaN = 尚无快照
        for col in depth_cols:
            df[col] = df[col].ffill()

    if depth_age is not None:
        df['depth_age_seconds'] = depth_age

    # 流量 / 计数 / 不平衡度：无成交秒自然为 0
    zero_fill_cols = [
        'volume', 'quote_volume',
        'trade_count', 'buy_count', 'sell_count',
        'buy_volume', 'sell_volume', 'buy_quote_volume', 'sell_quote_volume',
        'trade_intensity', 'avg_trade_size', 'max_trade_size', 'price_range',
        'tick_count', 'up_tick_count', 'down_tick_count',
        'volume_imbalance',
        'large_trade_count', 'large_trade_volume',
        'update_count',
    ]
    for col in zero_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df = df.dropna(axis=1, how='all')

    filepath = Path(settings.aggregated_data_dir) / f"{symbol}_{date_str}.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=True, date_format='%Y-%m-%d %H:%M:%S')
    if missing_count > 0:
        logger.info(f"补齐缺失秒: {missing_count} 条 (reindex 至严格连续)")
    logger.info(f"已保存: {filepath.name} ({len(df)} 条)")


class RealtimeSystem:
    """WebSocket 实时采集系统（回退模式）"""

    def __init__(self, symbol: str):
        self.settings = get_settings()
        self.symbol = symbol.lower()
        self.collector = None
        self.aggregator = None
        self.running = False
        self.market_data: list[dict] = []

    def on_message(self, stream_type: str, data: dict):
        try:
            if stream_type.startswith('depth'):
                parsed = BinanceCollector.parse_depth_update(data)
                self.aggregator.add_orderbook_update(parsed['symbol'], parsed)
            elif stream_type == 'trade':
                parsed = BinanceCollector.parse_trade(data)
                self.aggregator.add_trade(parsed['symbol'], parsed)
        except Exception as e:
            logger.error(f"消息处理错误: {e}", exc_info=True)

    def on_aggregated_data(self, data: UnifiedMarketData):
        row = unified_to_row(data)
        self.market_data.append(row)

        if len(self.market_data) >= 100:
            asyncio.create_task(self._save_batch())

    async def _save_batch(self):
        if not self.market_data:
            return
        try:
            df = pd.DataFrame(self.market_data).set_index('timestamp')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = Path(self.settings.aggregated_data_dir) / f"realtime_{self.symbol}_{timestamp}.csv"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(filepath, index=True)
            logger.info(f"已保存实时数据: {filepath.name} ({len(df)} 条)")
            self.market_data.clear()
        except Exception as e:
            logger.error(f"保存失败: {e}", exc_info=True)

    async def start(self):
        self.running = True
        logger.info(f"启动 WebSocket 实时采集: {self.symbol}")

        self.aggregator = SecondAggregator(on_aggregated_data=self.on_aggregated_data)
        await self.aggregator.start()

        self.collector = BinanceCollector(
            symbols=[self.symbol],
            streams=self.settings.streams,
            on_message=self.on_message
        )

        try:
            await self.collector.start()
        except KeyboardInterrupt:
            await self.stop()

    async def stop(self):
        if not self.running:
            return
        self.running = False
        if self.collector:
            await self.collector.stop()
        if self.aggregator:
            await self.aggregator.stop()
        if self.market_data:
            await self._save_batch()
        logger.info("实时采集已停止")


async def main():
    settings = get_settings()
    symbol = settings.symbols[0].upper()
    from datetime import date

    start_date = date(2026, 5, 12)
    end_date   = date(2026, 5, 13)

    logger.info("=" * 60)
    logger.info(f"Binance 数据采集: {symbol}")
    logger.info(f"日期范围: {start_date} ~ {end_date}")
    logger.info("优先历史数据，不可用时回退 WebSocket")
    logger.info("=" * 60)

    history = HistoryCollector(symbol)
    all_records: list[UnifiedMarketData] = []
    realtime_dates = []

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        records = history.collect_day(date_str)

        if records is not None:
            all_records.extend(records)
        else:
            logger.info(f"{date_str} 历史数据不可用，标记为实时采集")
            realtime_dates.append(date_str)

        current += timedelta(days=1)

    # 保存整个时间段为一个数据集
    if all_records:
        save_aggregated_data(all_records, symbol, f"{start_date}_{end_date}")

    if realtime_dates:
        logger.info(f"启动实时采集（覆盖日期: {realtime_dates}）")
        system = RealtimeSystem(symbol)
        try:
            await system.start()
        except Exception as e:
            logger.error(f"实时采集错误: {e}", exc_info=True)
        finally:
            await system.stop()
    else:
        logger.info("所有日期均已通过历史数据完成采集")


if __name__ == '__main__':
    asyncio.run(main())
