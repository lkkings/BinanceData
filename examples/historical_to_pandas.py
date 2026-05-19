"""把分钟级数据流加载到 pandas DataFrame 做分析

演示：
- 通过 HTTP NDJSON 流式接口拉取一段时间的数据
- 边接收边构造 pandas DataFrame（避免一次性加载到内存）
- 简单计算：分钟收益率、20 分钟移动均线、最大回撤

用法:
    uv run python examples/historical_to_pandas.py --symbol btcusdt --start 2026-05-12 --end 2026-05-15
"""
from __future__ import annotations

import argparse
import json
import sys

import httpx
import pandas as pd


def fetch_to_dataframe(
    base_url: str, symbol: str, start: str, end: str
) -> pd.DataFrame:
    """流式拉取并构造 DataFrame，时间索引为 UTC+8。"""
    rows: list[dict] = []
    with httpx.stream(
        "GET",
        f"{base_url}/api/range",
        params={"symbol": symbol, "start": start, "end": end, "chunk_size": 100},
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            msg = json.loads(line)
            records = msg.get("records") if isinstance(msg, dict) and "records" in msg else [msg]
            rows.extend(records)
            print(f"  ... received {len(rows)} records", end="\r", file=sys.stderr)

    print(file=sys.stderr)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    # 把价格类字段转成 float
    for col in ("open", "high", "low", "close", "vwap", "volume", "quote_volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def analyze(df: pd.DataFrame) -> None:
    if df.empty:
        print("[empty]")
        return

    df = df.sort_index()

    # 简单统计
    print("=" * 70)
    print(f"{'交易对':<14} {df['symbol'].iloc[0]}")
    print(f"{'数据范围':<14} {df.index[0]} ~ {df.index[-1]}")
    print(f"{'分钟数':<14} {len(df)}")
    print(f"{'起始价':<14} {df['open'].iloc[0]:.2f}")
    print(f"{'结束价':<14} {df['close'].iloc[-1]:.2f}")
    print(f"{'最高价':<14} {df['high'].max():.2f}")
    print(f"{'最低价':<14} {df['low'].min():.2f}")
    period_return = (df['close'].iloc[-1] / df['open'].iloc[0] - 1) * 100
    print(f"{'区间涨跌幅':<14} {period_return:+.3f}%")
    print(f"{'累计成交量':<14} {df['volume'].sum():.4f}")
    print(f"{'累计成交额':<14} {df['quote_volume'].sum():,.2f}")
    print(f"{'累计成交笔数':<14} {df['trade_count'].sum():,}")
    print()

    # 收益率分布
    df["ret"] = df["close"].pct_change()
    rets = df["ret"].dropna()
    print("分钟收益率分布")
    print(f"  均值      {rets.mean()*1e4:+.4f} bps")
    print(f"  标准差    {rets.std()*1e4:.4f} bps")
    print(f"  最大涨幅  {rets.max()*100:+.3f}%")
    print(f"  最大跌幅  {rets.min()*100:+.3f}%")
    print()

    # 20 分钟均线
    df["ma20"] = df["close"].rolling(20).mean()
    print("最近 5 条 close + MA20")
    tail = df[["close", "ma20"]].tail(5).copy()
    tail.columns = ["close", "MA20"]
    print(tail.to_string())
    print()

    # 最大回撤
    cummax = df["close"].cummax()
    drawdown = (df["close"] / cummax - 1)
    mdd_idx = drawdown.idxmin()
    print("最大回撤")
    print(f"  发生时刻  {mdd_idx}")
    print(f"  幅度      {drawdown.min()*100:+.3f}%")
    peak_idx = df["close"].loc[:mdd_idx].idxmax()
    print(f"  从峰值    {peak_idx} (close={df['close'].loc[peak_idx]:.2f})")
    print(f"  到谷值    {mdd_idx} (close={df['close'].loc[mdd_idx]:.2f})")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="把历史分钟数据加载到 pandas 做分析")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--symbol", required=True)
    p.add_argument("--start", required=True, help="UTC+8 起始 YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS")
    p.add_argument("--end", required=True, help="UTC+8 结束")
    p.add_argument("--save-csv", help="可选：另存为 CSV 文件")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[INFO] fetching {args.symbol} {args.start} ~ {args.end} ...", file=sys.stderr)
    df = fetch_to_dataframe(args.url, args.symbol, args.start, args.end)
    analyze(df)
    if args.save_csv and not df.empty:
        df.to_csv(args.save_csv)
        print(f"[INFO] saved to {args.save_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
