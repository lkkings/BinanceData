"""主程序入口 - Binance WebSocket 实时数据采集示例"""
import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.collectors import BinanceCollector
from src.aggregators import SecondAggregator
from src.config import get_settings
from src.models import UnifiedMarketData

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _to_float(value: Any) -> Any:
    """将 Decimal 安全转换为 float，保留 None。"""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return value


class DataCollectionSystem:
    """数据采集系统"""

    def __init__(self):
        self.settings = get_settings()
        self.collector = None
        self.aggregator = None
        self.running = False

        # 统一市场数据存储
        self.market_data: list[dict] = []

    def on_message(self, stream_type: str, data: dict):
        """WebSocket 消息回调

        Args:
            stream_type: 流类型，例如 "depth20@100ms", "trade", "bookTicker"
            data: 消息数据（已从组合流包装中提取）
        """
        try:
            if stream_type.startswith('depth'):
                parsed = BinanceCollector.parse_depth_update(data)
                self.aggregator.add_orderbook_update(parsed['symbol'], parsed)

            elif stream_type == 'trade':
                parsed = BinanceCollector.parse_trade(data)
                self.aggregator.add_trade(parsed['symbol'], parsed)

            elif stream_type == 'bookTicker':
                parsed = BinanceCollector.parse_book_ticker(data)
                logger.debug(f"BookTicker: {parsed['symbol']} "
                           f"Bid={parsed['best_bid_price']} "
                           f"Ask={parsed['best_ask_price']}")

        except KeyError as e:
            logger.error(f"消息字段缺失: {e}, stream_type={stream_type}, data={data}", exc_info=True)
        except Exception as e:
            logger.error(f"消息处理错误: {e}, stream_type={stream_type}", exc_info=True)

    def on_aggregated_data(self, data: UnifiedMarketData):
        """统一数据聚合完成回调"""
        # 概要日志
        ob_part = (
            f"价差={data.spread_bps:.2f}bps 不平衡={data.imbalance_5:.3f} "
            f"OB更新={data.update_count}"
            if data.spread_bps is not None else "OB=None"
        )
        tr_part = (
            f"收盘={float(data.close):.2f} 成交量={float(data.volume):.4f} "
            f"成交次数={data.trade_count}"
            if data.close is not None else "TR=None"
        )
        logger.info(
            f"[统一] {data.symbol} @ {data.timestamp.strftime('%H:%M:%S')} | "
            f"{ob_part} | {tr_part}"
        )

        # 将统一数据转换为 dict，Decimal -> float，便于 CSV 序列化
        row = {
            'timestamp': data.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': data.symbol,
            'best_bid_price': _to_float(data.best_bid_price),
            'best_bid_qty': _to_float(data.best_bid_qty),
            'best_ask_price': _to_float(data.best_ask_price),
            'best_ask_qty': _to_float(data.best_ask_qty),
            'spread_bps': data.spread_bps,
            'mid_price': _to_float(data.mid_price),
            'imbalance_5': data.imbalance_5,
            'update_count': data.update_count,
            'open': _to_float(data.open),
            'high': _to_float(data.high),
            'low': _to_float(data.low),
            'close': _to_float(data.close),
            'volume': _to_float(data.volume),
            'vwap': _to_float(data.vwap),
            'trade_count': data.trade_count,
            'buy_count': data.buy_count,
            'sell_count': data.sell_count,
            'buy_volume': _to_float(data.buy_volume),
            'sell_volume': _to_float(data.sell_volume),
        }

        self.market_data.append(row)

        # 每 100 条数据保存一次
        if len(self.market_data) >= 100:
            asyncio.create_task(self.save_market_data())

    async def save_market_data(self):
        """保存统一市场数据到 CSV 文件"""
        if not self.market_data:
            return

        try:
            import pandas as pd

            df = pd.DataFrame(self.market_data)

            # 列顺序固定，符合需求文档
            columns = [
                'timestamp', 'symbol',
                'best_bid_price', 'best_bid_qty',
                'best_ask_price', 'best_ask_qty',
                'spread_bps', 'mid_price', 'imbalance_5', 'update_count',
                'open', 'high', 'low', 'close',
                'volume', 'vwap',
                'trade_count', 'buy_count', 'sell_count',
                'buy_volume', 'sell_volume',
            ]
            df = df[columns]

            # timestamp 作为索引
            df = df.set_index('timestamp')

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"market_data_{timestamp}.csv"
            filepath = Path(self.settings.aggregated_data_dir) / filename

            filepath.parent.mkdir(parents=True, exist_ok=True)

            df.to_csv(filepath, index=True)

            logger.info(f"已保存统一市场数据: {filename} ({len(df)} 条记录)")

            self.market_data.clear()

        except Exception as e:
            logger.error(f"保存统一市场数据失败: {e}", exc_info=True)

    async def start(self):
        """启动数据采集系统"""
        self.running = True

        logger.info("=" * 60)
        logger.info("启动 Binance 数据采集系统")
        logger.info(f"交易对: {', '.join(self.settings.symbols)}")
        logger.info(f"数据流: {', '.join(self.settings.streams)}")
        logger.info(f"数据目录: {self.settings.aggregated_data_dir}")
        logger.info("=" * 60)

        self.aggregator = SecondAggregator(
            on_aggregated_data=self.on_aggregated_data
        )
        await self.aggregator.start()

        self.collector = BinanceCollector(
            symbols=self.settings.symbols,
            streams=self.settings.streams,
            on_message=self.on_message
        )

        try:
            await self.collector.start()
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止...")
            await self.stop()

    async def stop(self):
        """停止数据采集系统"""
        if not self.running:
            return

        self.running = False
        logger.info("正在停止数据采集系统...")

        if self.collector:
            await self.collector.stop()

        if self.aggregator:
            await self.aggregator.stop()

        if self.market_data:
            await self.save_market_data()

        logger.info("数据采集系统已停止")


async def main():
    """主函数"""
    system = DataCollectionSystem()

    try:
        await system.start()
    except Exception as e:
        logger.error(f"系统运行错误: {e}", exc_info=True)
    finally:
        await system.stop()


if __name__ == '__main__':
    asyncio.run(main())
