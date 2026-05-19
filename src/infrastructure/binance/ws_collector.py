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

    双连接模式：
    - Futures 端点：trade / depth（高频、合约市场独有）
    - Spot 端点：kline（futures kline 在某些网络不推送，spot 稳定且价格一致）
    """

    SPOT_WS_URL = "wss://stream.binance.com:9443"
    FUTURES_WS_URL = "wss://fstream.binance.com"

    def __init__(
        self,
        symbols: list[str],
        streams: list[str],
        is_futures: bool = False,
        on_message: Callable[[str, dict], None] | None = None
    ):
        self.symbols = [s.lower() for s in symbols]
        self.streams = streams
        self.is_futures = is_futures
        self.on_message = on_message

        self._primary_url = self.FUTURES_WS_URL if is_futures else self.SPOT_WS_URL
        self.running = False

        # kline 流不再订阅（kline 字段从 trade 流重建）
        self._effective_streams = [s for s in streams if not s.startswith("kline")]
        if len(self._effective_streams) != len(streams):
            dropped = [s for s in streams if s.startswith("kline")]
            logger.info(f"忽略 kline 流（kline 字段从 trade 重建）: {dropped}")

    def _build_stream_names(self, stream_list: list[str]) -> list[str]:
        result = []
        for symbol in self.symbols:
            for stream in stream_list:
                result.append(f"{symbol}@{stream}")
        return result

    async def _connect_ws(self, base_url: str, stream_names: list[str]) -> WebSocketClientProtocol:
        combined = "/".join(stream_names)
        url = f"{base_url}/stream?streams={combined}"
        logger.info(f"连接 WebSocket: {url}")
        ws = await websockets.connect(url, ping_interval=20, ping_timeout=10)
        logger.info(f"WebSocket 连接成功: {base_url}")
        return ws

    async def start(self):
        self.running = True
        if not self._effective_streams:
            logger.warning("没有可订阅的流（kline 流被忽略后为空）")
            return
        await self._run_loop("primary", self._primary_url, self._effective_streams)

    async def _run_loop(self, name: str, base_url: str, stream_list: list[str]):
        stream_names = self._build_stream_names(stream_list)
        while self.running:
            try:
                ws = await self._connect_ws(base_url, stream_names)
                try:
                    await self._receive_messages(ws)
                finally:
                    await ws.close()
            except websockets.ConnectionClosed:
                logger.warning(f"[{name}] WebSocket 连接关闭，5秒后重连...")
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{name}] 采集器错误: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _receive_messages(self, ws: WebSocketClientProtocol):
        async for message in ws:
            if not self.running:
                break
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
        例如：{"stream": "btcusdt@kline_1m", "data": {...}}
        """
        if "stream" not in data or "data" not in data:
            logger.warning(f"消息格式错误: {data}")
            return

        stream = data["stream"]
        payload = data["data"]

        # 从 stream name 提取 symbol 和 stream_type
        # 格式: btcusdt@trade / btcusdt@depth@100ms / btcusdt@kline_1m
        parts = stream.split("@")
        if len(parts) >= 2:
            symbol = parts[0]
            stream_type = "@".join(parts[1:])
            if stream_type.startswith("depth"):
                payload["symbol"] = symbol.upper()
        else:
            stream_type = stream

        if self.on_message:
            try:
                self.on_message(stream_type, payload)
            except Exception as e:
                logger.error(f"回调函数错误: {e}", exc_info=True)

    async def stop(self):
        self.running = False
        logger.info("WebSocket 采集器停止中...")

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
    def parse_trade(data: dict) -> dict | None:
        """解析成交数据（trade）

        Binance futures trade 流偶尔推送 X="NA", p="0", q="0" 的幽灵事件，
        这些不是真实成交，需要过滤。返回 None 表示应跳过。

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
            dict | None: 解析后的成交数据，或 None（幽灵事件）
        """
        if data.get("X") == "NA" or data.get("p") == "0" or data.get("q") == "0":
            return None

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
