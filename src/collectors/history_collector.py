"""Binance 历史逐笔成交数据下载与聚合"""
import logging
import zipfile
from decimal import Decimal
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import requests

from ..config import get_settings
from ..models import UnifiedMarketData

logger = logging.getLogger(__name__)


BOOK_DEPTH_BASE_URL = "https://data.binance.vision/data/futures/um/daily/bookDepth"
KLINES_BASE_URL = "https://data.binance.vision/data/futures/um/daily/klines"


class HistoryCollector:
    """从 Binance 官方历史数据下载逐笔成交、订单簿深度和K线并按秒聚合"""

    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.settings = get_settings()

    def _trade_zip_path(self, date_str: str) -> Path:
        return self.settings.raw_data_dir / f"{self.symbol}-trades-{date_str}.zip"

    def _depth_zip_path(self, date_str: str) -> Path:
        return self.settings.raw_data_dir / f"{self.symbol}-bookDepth-{date_str}.zip"

    def _klines_zip_path(self, date_str: str, interval: str = "1m") -> Path:
        return self.settings.raw_data_dir / f"{self.symbol}-{interval}-{date_str}.zip"

    def _download_file(self, url: str, local_path: Path) -> bool:
        if local_path.exists():
            logger.info(f"缓存命中: {local_path.name}")
            return True

        logger.info(f"下载: {url}")
        resp = requests.get(url, stream=
                            True, timeout=60)
        if resp.status_code != 200:
            logger.warning(f"下载失败 ({resp.status_code}): {url}")
            return False

        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

        logger.info(f"已下载: {local_path.name}")
        return True

    def download_day(self, date_str: str) -> Path | None:
        """下载某天的逐笔成交数据，返回 zip 路径或 None"""
        zip_path = self._trade_zip_path(date_str)
        url = f"{self.settings.history_base_url}/{self.symbol}/{self.symbol}-trades-{date_str}.zip"
        if self._download_file(url, zip_path):
            return zip_path
        return None

    def download_depth(self, date_str: str) -> Path | None:
        """下载某天的订单簿深度数据，返回 zip 路径或 None"""
        zip_path = self._depth_zip_path(date_str)
        url = f"{BOOK_DEPTH_BASE_URL}/{self.symbol}/{self.symbol}-bookDepth-{date_str}.zip"
        if self._download_file(url, zip_path):
            return zip_path
        return None

    def download_klines(self, date_str: str, interval: str = "1m") -> Path | None:
        """下载某天的K线数据，返回 zip 路径或 None"""
        zip_path = self._klines_zip_path(date_str, interval)
        url = f"{KLINES_BASE_URL}/{self.symbol}/{interval}/{self.symbol}-{interval}-{date_str}.zip"
        if self._download_file(url, zip_path):
            return zip_path
        return None

    def load_trades(self, zip_path: Path) -> pd.DataFrame:
        """从 zip 中读取逐笔成交数据

        仅过滤结构性无效数据（price/qty/quote_qty <= 0）和极端价格异常（滚动中位数 ±3%）。
        不做 qty 上界截断——大单对后续 `large_trade_*` 和 `max_trade_size` 等尾部特征有重要意义。
        """
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f)

        df.columns = ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker"]
        df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
        df = df.sort_values('time').reset_index(drop=True)

        original_len = len(df)

        df = df[(df['price'] > 0) & (df['qty'] > 0) & (df['quote_qty'] > 0)]

        rolling_median = df['price'].rolling(window=1000, center=True, min_periods=100).median()
        price_deviation = (df['price'] - rolling_median).abs() / rolling_median
        df = df[price_deviation < 0.03]

        df = df.reset_index(drop=True)
        removed = original_len - len(df)
        if removed > 0:
            logger.info(f"已过滤 {removed} 条结构性异常 ({removed/original_len*100:.3f}%)")

        return df

    def load_depth(self, zip_path: Path) -> pd.DataFrame:
        """从 zip 中读取订单簿深度数据，并过滤异常值

        数据格式: timestamp, percentage, depth, notional
        percentage 负值为 bid 侧，正值为 ask 侧
        每 ~30 秒一个快照，包含 ±0.2%~±5% 各档位
        """
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f)

        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        original_len = len(df)

        # 过滤无效数据
        df = df[(df['depth'] >= 0) & (df['notional'] >= 0)]

        # 深度异常值：按档位分组，超过各档位 99.5 分位数的视为异常
        valid_mask = pd.Series(True, index=df.index)
        for pct, group in df.groupby('percentage'):
            threshold = group['depth'].quantile(0.995)
            valid_mask.loc[group.index] = group['depth'] <= threshold
        df = df[valid_mask]

        df = df.reset_index(drop=True)
        removed = original_len - len(df)
        if removed > 0:
            logger.info(f"bookDepth 已过滤 {removed} 条异常数据 ({removed/original_len*100:.3f}%)")

        return df

    def load_klines(self, zip_path: Path) -> pd.DataFrame:
        """从 zip 中读取K线数据

        字段: open_time, open, high, low, close, volume, close_time,
              quote_volume, count, taker_buy_volume, taker_buy_quote_volume, ignore
        """
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f, header=None, skiprows=1)

        df.columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore"
        ]
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df = df.drop(columns=['close_time', 'ignore'])
        df = df.set_index('open_time').sort_index()
        return df

    def _compute_kline_features(self, klines_df: pd.DataFrame) -> pd.DataFrame:
        """从K线数据计算分钟级特征

        提取特征:
        - kline_open/high/low/close: 分钟K线 OHLC
        - kline_volume: 分钟成交量
        - kline_quote_volume: 分钟成交额
        - kline_count: 分钟成交笔数
        - kline_taker_buy_ratio: 主动买入占比 (taker_buy_volume / volume)
        - kline_body_ratio: 实体占比 |close - open| / (high - low)
        - kline_upper_shadow: 上影线比例
        - kline_lower_shadow: 下影线比例
        """
        df = klines_df.copy()

        df['kline_taker_buy_ratio'] = df['taker_buy_volume'] / df['volume'].replace(0, np.nan)
        df['kline_taker_buy_ratio'] = df['kline_taker_buy_ratio'].fillna(0.5)

        hl_range = df['high'] - df['low']
        body = (df['close'] - df['open']).abs()
        df['kline_body_ratio'] = body / hl_range.replace(0, np.nan)
        df['kline_body_ratio'] = df['kline_body_ratio'].fillna(0)

        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        df['kline_upper_shadow'] = upper_shadow / hl_range.replace(0, np.nan)
        df['kline_upper_shadow'] = df['kline_upper_shadow'].fillna(0)
        df['kline_lower_shadow'] = lower_shadow / hl_range.replace(0, np.nan)
        df['kline_lower_shadow'] = df['kline_lower_shadow'].fillna(0)

        df = df.rename(columns={
            'open': 'kline_open',
            'high': 'kline_high',
            'low': 'kline_low',
            'close': 'kline_close',
            'volume': 'kline_volume',
            'quote_volume': 'kline_quote_volume',
            'count': 'kline_count',
            'taker_buy_volume': 'kline_taker_buy_volume',
            'taker_buy_quote_volume': 'kline_taker_buy_quote_volume',
        })

        return df

    def _compute_depth_features(self, depth_df: pd.DataFrame) -> pd.DataFrame:
        """从 bookDepth 数据计算订单簿特征。向量化实现，按时间戳返回宽表"""
        if depth_df.empty:
            return pd.DataFrame()

        wide = depth_df.pivot_table(
            index='timestamp', columns='percentage', values='depth', aggfunc='last'
        )
        notional_wide = depth_df.pivot_table(
            index='timestamp', columns='percentage', values='notional', aggfunc='last'
        )

        out = pd.DataFrame(index=wide.index)

        def col(frame: pd.DataFrame, pct: float) -> pd.Series:
            return frame[pct] if pct in frame.columns else pd.Series(0.0, index=frame.index)

        out['bid_depth_02'] = col(wide, -0.2)
        out['ask_depth_02'] = col(wide, 0.2)
        out['bid_notional_02'] = col(notional_wide, -0.2)
        out['ask_notional_02'] = col(notional_wide, 0.2)
        out['bid_depth_1'] = col(wide, -1.0)
        out['ask_depth_1'] = col(wide, 1.0)

        bid_cols = [c for c in wide.columns if c < 0]
        ask_cols = [c for c in wide.columns if c > 0]
        out['total_bid_depth'] = wide[bid_cols].sum(axis=1) if bid_cols else 0.0
        out['total_ask_depth'] = wide[ask_cols].sum(axis=1) if ask_cols else 0.0

        def imbalance(bid: pd.Series, ask: pd.Series) -> pd.Series:
            total = bid + ask
            return np.where(total > 0, (bid - ask) / total, 0.0)

        out['depth_imbalance_02'] = imbalance(out['bid_depth_02'], out['ask_depth_02'])
        out['depth_imbalance_1'] = imbalance(out['bid_depth_1'], out['ask_depth_1'])
        out['depth_imbalance_total'] = imbalance(out['total_bid_depth'], out['total_ask_depth'])

        return out.sort_index()

    def aggregate_to_seconds(
        self, trades_df: pd.DataFrame, depth_df: pd.DataFrame, klines_df: pd.DataFrame | None = None
    ) -> Generator[UnifiedMarketData, None, None]:
        """将逐笔成交按秒聚合，合并订单簿深度和K线特征

        使用 merge_asof 做 O(N+M) 对齐（LOCF），替代每秒重扫的 O(N·M) 实现。
        """
        trades_df['second'] = trades_df['time'].dt.floor('1s')

        depth_features = self._compute_depth_features(depth_df)
        kline_features = None
        if klines_df is not None and not klines_df.empty:
            kline_features = self._compute_kline_features(klines_df)

        seconds = pd.Series(sorted(trades_df['second'].unique()), name='second')
        seconds_idx = pd.DataFrame({'second': seconds.astype('datetime64[ns, UTC]')})

        depth_lookup = None
        if not depth_features.empty:
            depth_frame = depth_features.reset_index().rename(columns={'timestamp': 'second'}).sort_values('second')
            depth_frame['second'] = depth_frame['second'].astype('datetime64[ns, UTC]')
            depth_lookup = pd.merge_asof(
                seconds_idx, depth_frame, on='second', direction='backward'
            ).set_index('second')

        kline_lookup = None
        if kline_features is not None and not kline_features.empty:
            kline_frame = kline_features.reset_index().rename(columns={'open_time': 'second'}).sort_values('second')
            kline_frame['second'] = kline_frame['second'].astype('datetime64[ns, UTC]')
            kline_lookup = pd.merge_asof(
                seconds_idx, kline_frame, on='second', direction='backward'
            ).set_index('second')

        for ts, group in trades_df.groupby('second'):
            ts_key = pd.Timestamp(ts).tz_convert('UTC') if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts, tz='UTC')
            ts_key = ts_key.as_unit('ns')
            depth_row = depth_lookup.loc[ts_key] if depth_lookup is not None and ts_key in depth_lookup.index else None
            kline_row = kline_lookup.loc[ts_key] if kline_lookup is not None and ts_key in kline_lookup.index else None

            if depth_row is not None and depth_row.isna().all():
                depth_row = None
            if kline_row is not None and kline_row.isna().all():
                kline_row = None

            yield self._compute_features(ts, group, depth_row, kline_row)

    def _compute_features(
        self, timestamp: pd.Timestamp, trades: pd.DataFrame,
        depth_row: pd.Series | None = None, kline_row: pd.Series | None = None
    ) -> UnifiedMarketData:
        """计算单秒内的所有特征"""
        prices = trades['price'].values
        qtys = trades['qty'].values
        quote_qtys = trades['quote_qty'].values
        is_buyer_maker = trades['is_buyer_maker'].values

        # OHLCV
        open_p = Decimal(str(prices[0]))
        high_p = Decimal(str(prices.max()))
        low_p = Decimal(str(prices.min()))
        close_p = Decimal(str(prices[-1]))
        volume = Decimal(str(qtys.sum()))
        quote_volume = Decimal(str(quote_qtys.sum()))
        # vwap: volume==0 不会出现在此函数中（trades 非空即至少一笔），保留除法；
        # 整秒零成交的补齐由 save_aggregated_data 完成
        vwap = Decimal(str(quote_qtys.sum() / qtys.sum()))

        # 买卖统计
        buy_mask = ~is_buyer_maker
        sell_mask = is_buyer_maker
        buy_vol = Decimal(str(qtys[buy_mask].sum()))
        sell_vol = Decimal(str(qtys[sell_mask].sum()))
        buy_quote_vol = Decimal(str(quote_qtys[buy_mask].sum()))
        sell_quote_vol = Decimal(str(quote_qtys[sell_mask].sum()))
        buy_count = int(buy_mask.sum())
        sell_count = int(sell_mask.sum())

        # 高频特征
        trade_count = len(trades)
        trade_intensity = float(trade_count)
        avg_trade_size = Decimal(str(qtys.mean()))
        max_trade_size = Decimal(str(qtys.max()))
        price_range = high_p - low_p

        # tick 统计
        price_diffs = np.diff(prices)
        tick_count = int(np.count_nonzero(price_diffs))
        up_tick_count = int((price_diffs > 0).sum())
        down_tick_count = int((price_diffs < 0).sum())

        # 量不平衡
        total_vol = float(volume)
        vol_imbalance = (float(buy_vol) - float(sell_vol)) / total_vol if total_vol > 0 else 0.0

        # 大单检测 (> mean + 2*std)
        if len(qtys) > 1:
            threshold = qtys.mean() + 2 * qtys.std()
            large_mask = qtys > threshold
            large_count = int(large_mask.sum())
            large_volume = Decimal(str(qtys[large_mask].sum()))
        else:
            large_count = 0
            large_volume = Decimal('0')

        return UnifiedMarketData(
            timestamp=timestamp.to_pydatetime(),
            symbol=self.symbol,
            open=open_p,
            high=high_p,
            low=low_p,
            close=close_p,
            volume=volume,
            quote_volume=quote_volume,
            vwap=vwap,
            trade_count=trade_count,
            buy_count=buy_count,
            sell_count=sell_count,
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            buy_quote_volume=buy_quote_vol,
            sell_quote_volume=sell_quote_vol,
            trade_intensity=trade_intensity,
            avg_trade_size=avg_trade_size,
            max_trade_size=max_trade_size,
            price_range=price_range,
            tick_count=tick_count,
            up_tick_count=up_tick_count,
            down_tick_count=down_tick_count,
            volume_imbalance=vol_imbalance,
            large_trade_count=large_count,
            large_trade_volume=large_volume,
            # 订单簿深度特征
            bid_depth_02=float(depth_row['bid_depth_02']) if depth_row is not None and 'bid_depth_02' in depth_row else None,
            ask_depth_02=float(depth_row['ask_depth_02']) if depth_row is not None and 'ask_depth_02' in depth_row else None,
            bid_notional_02=float(depth_row['bid_notional_02']) if depth_row is not None and 'bid_notional_02' in depth_row else None,
            ask_notional_02=float(depth_row['ask_notional_02']) if depth_row is not None and 'ask_notional_02' in depth_row else None,
            depth_imbalance_02=float(depth_row['depth_imbalance_02']) if depth_row is not None and 'depth_imbalance_02' in depth_row else None,
            bid_depth_1=float(depth_row['bid_depth_1']) if depth_row is not None and 'bid_depth_1' in depth_row else None,
            ask_depth_1=float(depth_row['ask_depth_1']) if depth_row is not None and 'ask_depth_1' in depth_row else None,
            depth_imbalance_1=float(depth_row['depth_imbalance_1']) if depth_row is not None and 'depth_imbalance_1' in depth_row else None,
            total_bid_depth=float(depth_row['total_bid_depth']) if depth_row is not None and 'total_bid_depth' in depth_row else None,
            total_ask_depth=float(depth_row['total_ask_depth']) if depth_row is not None and 'total_ask_depth' in depth_row else None,
            depth_imbalance_total=float(depth_row['depth_imbalance_total']) if depth_row is not None and 'depth_imbalance_total' in depth_row else None,
            # K线特征
            kline_open=float(kline_row['kline_open']) if kline_row is not None else None,
            kline_high=float(kline_row['kline_high']) if kline_row is not None else None,
            kline_low=float(kline_row['kline_low']) if kline_row is not None else None,
            kline_close=float(kline_row['kline_close']) if kline_row is not None else None,
            kline_volume=float(kline_row['kline_volume']) if kline_row is not None else None,
            kline_quote_volume=float(kline_row['kline_quote_volume']) if kline_row is not None else None,
            kline_count=int(kline_row['kline_count']) if kline_row is not None else None,
            kline_taker_buy_volume=float(kline_row['kline_taker_buy_volume']) if kline_row is not None else None,
            kline_taker_buy_quote_volume=float(kline_row['kline_taker_buy_quote_volume']) if kline_row is not None else None,
            kline_taker_buy_ratio=float(kline_row['kline_taker_buy_ratio']) if kline_row is not None else None,
            kline_body_ratio=float(kline_row['kline_body_ratio']) if kline_row is not None else None,
            kline_upper_shadow=float(kline_row['kline_upper_shadow']) if kline_row is not None else None,
            kline_lower_shadow=float(kline_row['kline_lower_shadow']) if kline_row is not None else None,
        )

    def collect_day(self, date_str: str) -> list[UnifiedMarketData] | None:
        """下载并聚合某天的数据（trades + bookDepth + klines），返回 None 表示数据不可用"""
        logger.info(f"聚合 {self.symbol} {date_str} ...")

        zip_path = self.download_day(date_str)
        if zip_path is None:
            return None
        trades_df = self.load_trades(zip_path)
        logger.info(f"已加载逐笔成交: {len(trades_df)} 条")

        depth_zip = self.download_depth(date_str)
        depth_df = self.load_depth(depth_zip) if depth_zip else pd.DataFrame()
        if not depth_df.empty:
            logger.info(f"已加载订单簿深度: {len(depth_df)} 条")

        klines_zip = self.download_klines(date_str)
        klines_df = self.load_klines(klines_zip) if klines_zip else None
        if klines_df is not None:
            logger.info(f"已加载K线: {len(klines_df)} 条")

        results = list(self.aggregate_to_seconds(trades_df, depth_df, klines_df))
        logger.info(f"完成: {len(results)} 条秒级数据")
        return results
