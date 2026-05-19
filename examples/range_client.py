"""HTTP 区间查询客户端示例

按时间范围流式拉取分钟级数据，时间用 UTC+8 字符串。

用法:
    uv run python examples/range_client.py --symbol btcusdt --start 2026-05-12 --end 2026-05-15
    uv run python examples/range_client.py --symbol btcusdt --start "2026-05-15 09:00:00" --end "2026-05-15 18:00:00"
    uv run python examples/range_client.py --symbol btcusdt --start 2026-05-12 --end 2026-05-15 --output data.ndjson
"""
from __future__ import annotations

import argparse
import json
import sys

import httpx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按时间区间拉取分钟级市场数据 (NDJSON 流式)")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--symbol", required=True, help="交易对，例如 btcusdt")
    p.add_argument("--start", required=True, help="UTC+8 起始时间，YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS")
    p.add_argument("--end", required=True, help="UTC+8 结束时间，同 start 格式")
    p.add_argument("--chunk-size", type=int, default=1, help="每个流式块包含的记录数（默认 1）")
    p.add_argument("--output", help="输出文件路径（默认打印到 stdout，每行一个 JSON）")
    p.add_argument("--summary", action="store_true", help="只打印汇总统计，不打印每条记录")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    params = {
        "symbol": args.symbol,
        "start": args.start,
        "end": args.end,
        "chunk_size": args.chunk_size,
    }

    out_file = open(args.output, "w") if args.output else None
    count = 0
    first_ts: str | None = None
    last_ts: str | None = None

    try:
        with httpx.stream(
            "GET",
            f"{args.url}/api/range",
            params=params,
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
        ) as resp:
            if resp.status_code != 200:
                resp.read()
                print(f"[ERROR] HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
                sys.exit(1)

            print(
                f"[INFO] streaming from {resp.headers.get('X-Start-CST')} "
                f"to {resp.headers.get('X-End-CST')} (UTC+8)...",
                file=sys.stderr,
            )

            for line in resp.iter_lines():
                if not line:
                    continue
                msg = json.loads(line)

                # 处理 chunk_size>1 的批量格式
                records = msg.get("records") if isinstance(msg, dict) and "records" in msg else [msg]

                for r in records:
                    count += 1
                    ts = r.get("timestamp")
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts

                    if not args.summary:
                        if out_file:
                            out_file.write(json.dumps(r, ensure_ascii=False) + "\n")
                        else:
                            print(
                                f"{ts}  {r.get('symbol')}  "
                                f"O={r.get('open')}  H={r.get('high')}  "
                                f"L={r.get('low')}  C={r.get('close')}  "
                                f"V={r.get('volume')}  trades={r.get('trade_count')}"
                            )
    finally:
        if out_file:
            out_file.close()

    print(
        f"\n[DONE] {count} records  range: {first_ts} ~ {last_ts}",
        file=sys.stderr,
    )
    if args.output:
        print(f"[DONE] saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
