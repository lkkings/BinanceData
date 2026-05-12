"""Binance WebSocket 数据采集器"""
import asyncio
import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Callable, Any

import websockets
from websockets.client import WebSocketClientProtocol

logger = logging.getLogger(__name__)


class BinanceCollector:
    """Binance WebSocket 实时数据采集器

    支持的数据流：
    - depth@100ms: L2 订单簿增量更新
    - trade: 逐笔成交
    - bookTicker: 最优买卖价
    - ticker: 24小时行情
    - forceOrder: 强平订单（仅合约）
    """

    # Binance WebSocket 端点
    # 注意：组合流使用 /stream?streams= 格式，会包装消息为 {"stream": "...", "data": {...}}
    SPOT_WS_URL = "wss://stream.binance.com:9443"
    FUTURES_WS_URL = "wss://fstream.binance.com"

    def __init__(
        self,
        symbols: list[str],
        streams: list[str],
        is_futures: bool = False,
        on_message: Callable[[str, dict], None] | None = None
    ):
        """初始化采集器

        Args:
            symbols: 交易对列表，如 ['btcusdt', 'ethusdt']
            streams: 数据流列表，如 ['depth@100ms', 'trade', 'bookTicker']
            is_futures: 是否为合约市场
            on_message: 消息回调函数 callback(stream_name, data)
        """
        self.symbols = [s.lower() for s in symbols]
        self.streams = streams
        self.is_futures = is_futures
        self.on_message = on_message

        self.ws_url = self.FUTURES_WS_URL if is_futures else self.SPOT_WS_URL
        self.ws: WebSocketClientProtocol | None = None
        self.running = False

    def _build_stream_names(self) -> list[str]:
        """构建完整的流名称"""
        stream_names = []
        for symbol in self.symbols:
            for stream in self.streams:
                stream_names.append(f"{symbol}@{stream}")
        return stream_names

    async def connect(self):
        """建立 WebSocket 连接（使用组合流格式）"""
        stream_names = self._build_stream_names()
        combined_streams = "/".join(stream_names)
        # 使用组合流格式：/stream?streams=xxx，消息会被包装为 {"stream": "...", "data": {...}}
        url = f"{self.ws_url}/stream?streams={combined_streams}"

        logger.info(f"连接到 Binance WebSocket: {url}")
        self.ws = await websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10
        )
        logger.info("WebSocket 连接成功")

    async def start(self):
        """启动数据采集"""
        self.running = True

        while self.running:
            try:
                await self.connect()
                await self._receive_messages()
            except websockets.ConnectionClosed:
                logger.warning("WebSocket 连接关闭，5秒后重连...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"采集器错误: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _receive_messages(self):
        """接收并处理消息"""
        if not self.ws:
            return

        async for message in self.ws:
            try:
                data = json.loads(message)
                await self._handle_message(data)
            except json.JSONDecodeError as e:
                logger.error(f"JSON 解析错误: {e}")
            except Exception as e:
                logger.error(f"消息处理错误: {e}", exc_info=True)

    async def _handle_message(self, data: dict):
        """处理单条消息

        组合流消息格式：{"stream": "<streamName>", "data": <rawPayload>}
        例如：{"stream": "btcusdt@depth20@100ms", "data": {...}}
        """
        if "stream" not in data or "data" not in data:
            logger.warning(f"消息格式错误，缺少 stream 或 data 字段: {data}")
            return
        
        stream = data["stream"]
        payload = data["data"]
        
        # 提取流类型（例如从 "btcusdt@depth20@100ms" 提取 "depth20@100ms"）
        # 对于 depth 流，我们需要提取完整的流类型以便识别
        parts = stream.split("@")
        if len(parts) >= 2:
            # 提取交易对和流类型
            symbol = parts[0]
            stream_type = "@".join(parts[1:])  # 例如 "depth20@100ms" 或 "trade"
            # 为 depth 流添加 symbol 信息（因为 depth 消息本身不包含 symbol）
            if stream_type.startswith("depth"):
                payload["symbol"] = symbol.upper()
        else:
            stream_type = stream

        # 调用回调函数
        if self.on_message:
            try:
                self.on_message(stream_type, payload)
            except Exception as e:
                logger.error(f"回调函数错误: {e}", exc_info=True)

    async def stop(self):
        """停止采集"""
        self.running = False
        if self.ws:
            await self.ws.close()
            logger.info("WebSocket 连接已关闭")

    @staticmethod
    def parse_depth_update(data: dict) -> dict:
        """解析订单簿增量更新（depthUpdate）

        根据 Binance 官方文档，depth@100ms 流的消息格式为：
        {
            "e": "depthUpdate",
            "E": 1778573295814,
            "s": "BTCUSDT",
            "U": 93547321531,
            "u": 93547321568,
            "b": [["80892.03000000", "6.08688000"], ...],
            "a": [["80892.04000000", "0.57286000"], ...],
            "symbol": "BTCUSDT"  # 由 _handle_message 添加
        }

        Returns:
            {
                'event_time': int,
                'symbol': str,
                'first_update_id': int,
                'final_update_id': int,
                'bids': [(price, qty), ...],
                'asks': [(price, qty), ...]
            }
        """
        return {
            'event_time': data['E'],
            'symbol': data['s'],
            'first_update_id': data['U'],
            'final_update_id': data['u'],
            'bids': [(Decimal(p), Decimal(q)) for p, q in data['b']],
            'asks': [(Decimal(p), Decimal(q)) for p, q in data['a']]
        }

    @staticmethod
    def parse_trade(data: dict) -> dict:
        """解析成交数据（trade）

        根据 Binance 官方文档，trade 流的消息格式为：
        {
            "e": "trade",
            "E": 1778573295883,
            "s": "ETHUSDT",
            "t": 4009727232,
            "p": "2290.53000000",
            "q": "0.05060000",
            "T": 1778573295883,
            "m": true,
            "M": true
        }

        Returns:
            {
                'event_time': int,
                'symbol': str,
                'trade_id': int,
                'price': Decimal,
                'quantity': Decimal,
                'trade_time': int,
                'is_buyer_maker': bool
            }
        """
        return {
            'event_time': data['E'],
            'symbol': data['s'],
            'trade_id': data['t'],
            'price': Decimal(data['p']),
            'quantity': Decimal(data['q']),
            'trade_time': data['T'],
            'is_buyer_maker': data['m']
        }

    @staticmethod
    def parse_book_ticker(data: dict) -> dict:
        """解析最优买卖价

        Returns:
            {
                'update_id': int,
                'symbol': str,
                'best_bid_price': Decimal,
                'best_bid_qty': Decimal,
                'best_ask_price': Decimal,
                'best_ask_qty': Decimal
            }
        """
        return {
            'update_id': data['u'],
            'symbol': data['s'],
            'best_bid_price': Decimal(data['b']),
            'best_bid_qty': Decimal(data['B']),
            'best_ask_price': Decimal(data['a']),
            'best_ask_qty': Decimal(data['A'])
        }
