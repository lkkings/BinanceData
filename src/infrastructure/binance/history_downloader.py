"""Binance 历史逐笔成交数据下载与分块流式聚合

内存安全设计：
- trade CSV 分块读取（100K 行/块 ≈ 5MB），不全量加载
- 同一分钟的数据可能跨越两个 chunk，通过 minute_buffer 正确合并
- depth 数据较小（~28K 行/天 ≈ 2MB），全量加载
- 峰值内存 ≈ 10MB，适合 500MB 内存机器
"""
import logging
import zipfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import requests

from ...config import get_settings
from ...domain import UnifiedMarketData

logger = logging.getLogger(__name__)

BOOK_DEPTH_BASE_URL = "https://data.binance.vision/data/futures/um/daily/bookDepth"


class HistoryDownloader:
    """从 Binance 官方历史数据下载逐笔成交、订单簿深度并聚合到分钟"""

    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.settings = get_settings()

    # ─── 下载 ───────────────────────────────────────────────────────────

    def _trade_zip_path(self, date_str: str) -> Path:
        return self.settings.raw_data_dir / f"{self.symbol}-trades-{date_str}.zip"

    def _depth_zip_path(self, date_str: str) -> Path:
        return self.settings.raw_data_dir / f"{self.symbol}-bookDepth-{date_str}.zip"

    def _download_file(self, url: str, local_path: Path) -> bool:
        if local_path.exists():
            return True
        logger.info(f"下载: {url}")
        resp = requests.get(url, stream=True, timeout=60)
        if resp.status_code != 200:
            logger.warning(f"下载失败 ({resp.status_code}): {url}")
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        return True

    def download_day(self, date_str: str) -> Path | None:
        zip_path = self._trade_zip_path(date_str)
        url = f"{self.settings.history_base_url}/{self.symbol}/{self.symbol}-trades-{date_str}.zip"
        if self._download_file(url, zip_path):
            return zip_path
        return None

    def download_depth(self, date_str: str) -> Path | None:
        zip_path = self._depth_zip_path(date_str)
        url = f"{BOOK_DEPTH_BASE_URL}/{self.symbol}/{self.symbol}-bookDepth-{date_str}.zip"
        if self._download_file(url, zip_path):
            return zip_path
        return None

    # ─── 分块读取 ────────────────────────────────────────────────────────

    def _iter_trade_chunks(
        self, zip_path: Path, chunksize: int = 100_000
    ) -> Generator[pd.DataFrame, None, None]:
        """分块流式读取 trade CSV。

        Binance 历史 CSV 已按时间排序。每块 100K 行 ≈ 5MB。
        同一分钟的数据可能跨越两个 chunk，由调用方通过 buffer 合并。
        """
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                reader = pd.read_csv(
                    f,
                    chunksize=chunksize,
                    names=["id", "price", "qty", "quote_qty", "time", "is_buyer_maker"],
                    header=0,
                )
                for chunk in reader:
                    chunk = chunk[
                        (chunk['price'] > 0) & (chunk['qty'] > 0) & (chunk['quote_qty'] > 0)
                    ]
                    if chunk.empty:
                        continue
                    chunk['time'] = pd.to_datetime(chunk['time'], unit='ms', utc=True)
                    yield chunk.reset_index(drop=True)

    def load_depth(self, zip_path: Path) -> pd.DataFrame:
        """读取订单簿深度数据（~28K 行/天 ≈ 2MB，可全量加载）"""
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df[(df['depth'] >= 0) & (df['notional'] >= 0)]
        return df.reset_index(drop=True)

    # ─── 深度特征 ────────────────────────────────────────────────────────

    def _compute_depth_features(self, depth_df: pd.DataFrame) -> pd.DataFrame:
        if depth_df.empty:
            return pd.DataFrame()

        wide = depth_df.pivot_table(
            index='timestamp', columns='percentage', values='depth', aggfunc='last'
        )
        notional_wide = depth_df.pivot_table(
            index='timestamp', columns='percentage', values='notional', aggfunc='last'
        )

        out = pd.DataFrame(index=wide.index)

        def col(frame, pct):
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

        def imbalance(bid, ask):
            total = bid + ask
            return np.where(total > 0, (bid - ask) / total, 0.0)

        out['depth_imbalance_02'] = imbalance(out['bid_depth_02'], out['ask_depth_02'])
        out['depth_imbalance_1'] = imbalance(out['bid_depth_1'], out['ask_depth_1'])
        out['depth_imbalance_total'] = imbalance(out['total_bid_depth'], out['total_ask_depth'])

        return out.sort_index()

    def _lookup_depth(self, depth_features: pd.DataFrame, minute_ts) -> pd.Series | None:
        """LOCF 查找 <= minute_ts 的最近深度快照"""
        if depth_features.empty:
            return None
        ts = pd.Timestamp(minute_ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        else:
            ts = ts.tz_convert('UTC')
        mask = depth_features.index <= ts
        if not mask.any():
            return None
        row = depth_features.loc[mask].iloc[-1]
        return None if row.isna().all() else row

    # ─── 分块流式聚合 ────────────────────────────────────────────────────

    def _aggregate_chunked(
        self, zip_path: Path, depth_df: pd.DataFrame
    ) -> Generator[UnifiedMarketData, None, None]:
        """分块流式聚合：逐块读取 trade CSV，按分钟边界切割并聚合。

        关键：同一分钟的 trade 可能跨越两个 chunk。通过 minute_buffer
        缓冲当前分钟的所有片段，只在看到下一分钟时才 flush 上一分钟。
        """
        depth_features = self._compute_depth_features(depth_df)

        current_minute: pd.Timestamp | None = None
        minute_buffer: list[pd.DataFrame] = []

        for chunk in self._iter_trade_chunks(zip_path):
            chunk['minute'] = chunk['time'].dt.floor('1min')

            for minute_ts, group in chunk.groupby('minute', sort=True):
                if current_minute is None:
                    current_minute = minute_ts

                if minute_ts != current_minute:
                    merged = pd.concat(minute_buffer, ignore_index=True)
                    depth_row = self._lookup_depth(depth_features, current_minute)
                    yield self._compute_features(current_minute, merged, depth_row)
                    current_minute = minute_ts
                    minute_buffer = []

                minute_buffer.append(group)

        if minute_buffer and current_minute is not None:
            merged = pd.concat(minute_buffer, ignore_index=True)
            depth_row = self._lookup_depth(depth_features, current_minute)
            yield self._compute_features(current_minute, merged, depth_row)

    # ─── 特征计算 ────────────────────────────────────────────────────────

    def _compute_features(
        self, timestamp: pd.Timestamp, trades: pd.DataFrame,
        depth_row: pd.Series | None = None
    ) -> UnifiedMarketData:
        prices = trades['price'].values
        qtys = trades['qty'].values
        quote_qtys = trades['quote_qty'].values
        is_buyer_maker = trades['is_buyer_maker'].values

        open_p = Decimal(str(prices[0]))
        high_p = Decimal(str(prices.max()))
        low_p = Decimal(str(prices.min()))
        close_p = Decimal(str(prices[-1]))
        volume = Decimal(str(qtys.sum()))
        quote_volume = Decimal(str(quote_qtys.sum()))
        vwap = Decimal(str(quote_qtys.sum() / qtys.sum()))

        buy_mask = ~is_buyer_maker
        buy_vol = Decimal(str(qtys[buy_mask].sum()))
        sell_vol = Decimal(str(qtys[~buy_mask].sum()))
        buy_quote_vol = Decimal(str(quote_qtys[buy_mask].sum()))
        sell_quote_vol = Decimal(str(quote_qtys[~buy_mask].sum()))
        buy_count = int(buy_mask.sum())
        sell_count = int((~buy_mask).sum())

        trade_count = len(trades)
        avg_trade_size = Decimal(str(qtys.mean()))
        max_trade_size = Decimal(str(qtys.max()))
        price_range = high_p - low_p

        price_diffs = np.diff(prices)
        tick_count = int(np.count_nonzero(price_diffs))
        up_tick_count = int((price_diffs > 0).sum())
        down_tick_count = int((price_diffs < 0).sum())

        total_vol = float(volume)
        vol_imbalance = (float(buy_vol) - float(sell_vol)) / total_vol if total_vol > 0 else 0.0

        if len(qtys) > 1:
            threshold = qtys.mean() + 2 * qtys.std()
            large_mask = qtys > threshold
            large_count = int(large_mask.sum())
            large_volume = Decimal(str(qtys[large_mask].sum()))
        else:
            large_count = 0
            large_volume = Decimal('0')

        hl_range = high_p - low_p
        body = abs(close_p - open_p)
        kline_taker_buy_ratio = float(buy_vol / volume) if volume > 0 else 0.5
        kline_body_ratio = float(body / hl_range) if hl_range > 0 else 0.0
        kline_upper_shadow = float((high_p - max(open_p, close_p)) / hl_range) if hl_range > 0 else 0.0
        kline_lower_shadow = float((min(open_p, close_p) - low_p) / hl_range) if hl_range > 0 else 0.0

        def _d(name):
            return float(depth_row[name]) if depth_row is not None and name in depth_row else None

        return UnifiedMarketData(
            timestamp=timestamp.to_pydatetime(),
            symbol=self.symbol,
            open=open_p, high=high_p, low=low_p, close=close_p,
            volume=volume, quote_volume=quote_volume, vwap=vwap,
            trade_count=trade_count,
            buy_count=buy_count, sell_count=sell_count,
            buy_volume=buy_vol, sell_volume=sell_vol,
            buy_quote_volume=buy_quote_vol, sell_quote_volume=sell_quote_vol,
            trade_intensity=float(trade_count),
            avg_trade_size=avg_trade_size, max_trade_size=max_trade_size,
            price_range=price_range,
            tick_count=tick_count, up_tick_count=up_tick_count, down_tick_count=down_tick_count,
            volume_imbalance=vol_imbalance,
            large_trade_count=large_count, large_trade_volume=large_volume,
            bid_depth_02=_d('bid_depth_02'), ask_depth_02=_d('ask_depth_02'),
            bid_notional_02=_d('bid_notional_02'), ask_notional_02=_d('ask_notional_02'),
            depth_imbalance_02=_d('depth_imbalance_02'),
            bid_depth_1=_d('bid_depth_1'), ask_depth_1=_d('ask_depth_1'),
            depth_imbalance_1=_d('depth_imbalance_1'),
            total_bid_depth=_d('total_bid_depth'), total_ask_depth=_d('total_ask_depth'),
            depth_imbalance_total=_d('depth_imbalance_total'),
            kline_open=float(open_p), kline_high=float(high_p),
            kline_low=float(low_p), kline_close=float(close_p),
            kline_volume=float(volume), kline_quote_volume=float(quote_volume),
            kline_count=trade_count,
            kline_taker_buy_volume=float(buy_vol),
            kline_taker_buy_quote_volume=float(buy_quote_vol),
            kline_taker_buy_ratio=kline_taker_buy_ratio,
            kline_body_ratio=kline_body_ratio,
            kline_upper_shadow=kline_upper_shadow,
            kline_lower_shadow=kline_lower_shadow,
        )

    # ─── 公开接口 ────────────────────────────────────────────────────────

    def collect_day_minutes(self, date_str: str) -> list[UnifiedMarketData] | None:
        """下载并按分钟聚合某天的数据（分块流式，内存安全）。

        返回 None 表示 trade 数据不可用，[] 表示聚合后为空。
        成功时返回 1440 条（无成交分钟用 close ffill 填充）。
        """
        logger.info(f"聚合 {self.symbol} {date_str} (chunked) ...")

        zip_path = self.download_day(date_str)
        if zip_path is None:
            return None

        depth_zip = self.download_depth(date_str)
        depth_df = self.load_depth(depth_zip) if depth_zip else pd.DataFrame()
        if not depth_df.empty:
            logger.info(f"已加载订单簿深度: {len(depth_df)} 条")

        records = list(self._aggregate_chunked(zip_path, depth_df))
        if not records:
            logger.warning(f"{date_str} 聚合后为空")
            return []

        records = self._fill_missing_minutes(records, date_str)
        logger.info(f"完成: {len(records)} 条分钟数据")
        return records

    # ─── 缺失分钟填充 ────────────────────────────────────────────────────

    @staticmethod
    def _fill_missing_minutes(
        records: list[UnifiedMarketData], date_str: str
    ) -> list[UnifiedMarketData]:
        """对一天 1440 分钟做严格连续补齐：缺失分钟用上一分钟的 close 填充。"""
        if not records:
            return records

        day_start = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        existing = {r.timestamp: r for r in records}

        out: list[UnifiedMarketData] = []
        last: UnifiedMarketData | None = None
        zero = Decimal("0")

        for i in range(1440):
            ts = day_start + timedelta(minutes=i)
            if ts in existing:
                last = existing[ts]
                out.append(last)
                continue
            if last is None:
                continue

            close = last.close
            filled = replace(
                last,
                timestamp=ts,
                open=close, high=close, low=close, close=close,
                volume=zero, quote_volume=zero, vwap=close,
                trade_count=0, buy_count=0, sell_count=0,
                buy_volume=zero, sell_volume=zero,
                buy_quote_volume=zero, sell_quote_volume=zero,
                trade_intensity=0.0, avg_trade_size=zero, max_trade_size=zero,
                price_range=zero,
                tick_count=0, up_tick_count=0, down_tick_count=0,
                volume_imbalance=0.0,
                large_trade_count=0, large_trade_volume=zero,
                kline_open=float(close), kline_high=float(close),
                kline_low=float(close), kline_close=float(close),
                kline_volume=0.0, kline_quote_volume=0.0, kline_count=0,
                kline_taker_buy_volume=0.0, kline_taker_buy_quote_volume=0.0,
                kline_taker_buy_ratio=0.5, kline_body_ratio=0.0,
                kline_upper_shadow=0.0, kline_lower_shadow=0.0,
            )
            out.append(filled)

        if out and out[0].timestamp != day_start:
            first = out[0]
            close = first.close
            head: list[UnifiedMarketData] = []
            ts = day_start
            while ts < first.timestamp:
                head.append(replace(
                    first,
                    timestamp=ts,
                    open=close, high=close, low=close, close=close,
                    volume=zero, quote_volume=zero, vwap=close,
                    trade_count=0, buy_count=0, sell_count=0,
                    buy_volume=zero, sell_volume=zero,
                    buy_quote_volume=zero, sell_quote_volume=zero,
                    trade_intensity=0.0, avg_trade_size=zero, max_trade_size=zero,
                    price_range=zero,
                    tick_count=0, up_tick_count=0, down_tick_count=0,
                    volume_imbalance=0.0,
                    large_trade_count=0, large_trade_volume=zero,
                    kline_open=float(close), kline_high=float(close),
                    kline_low=float(close), kline_close=float(close),
                    kline_volume=0.0, kline_quote_volume=0.0, kline_count=0,
                    kline_taker_buy_volume=0.0, kline_taker_buy_quote_volume=0.0,
                    kline_taker_buy_ratio=0.5, kline_body_ratio=0.0,
                    kline_upper_shadow=0.0, kline_lower_shadow=0.0,
                ))
                ts += timedelta(minutes=1)
            out = head + out

        return out
