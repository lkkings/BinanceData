"""
BTCUSDT 永续合约高频做市/微趋势策略

策略类型: Maker-only 微趋势捕获 (Maker-Only Micro-Momentum)
数据频率: 1 秒聚合 K 线 + 盘口
手续费假设: maker = 0.02%(2bps), taker = 0.05%(5bps)
关键洞察: 1 分钟内典型价差 0.01 USD (~0.0013bps), 远小于手续费,
          单边 taker 进出场 = 10bps, 而 10s 内 |涨跌|>10bps 仅占 6%,
          因此**必须全程 maker** (挂单), 单边吃单将完全吞噬利润。

核心信号 (按相关性排序,与 5s 后收益 corr=0.60):
  1. imbalance_5         订单簿前 5 档失衡       (corr 0.48)
  2. kline_taker_buy_ratio  K线吃单买比          (corr 0.44)
  3. trade_intensity     成交强度                 (corr 0.40)
  4. depth_imbalance_02  ±0.2% 深度失衡           (corr 0.36)
  5. volume_imbalance    买卖量失衡               (corr 0.32)

风险控制:
  - 单笔风险 0.5%, 最大持仓时间 30s
  - 止损 = 8 bps (回撤 4 倍 maker 往返成本)
  - 止盈 = 12 bps (3:1 盈亏比扣除 maker 往返 4 bps 后净赚 8 bps)
  - 信号阈值 = 分位 80/20, 避免低质量信号
  - 价差 > 2 USD 暂停交易 (流动性恶化)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass


MAKER_FEE_BPS = 2.0   # 0.02%
TAKER_FEE_BPS = 5.0   # 0.05%

# 策略参数 (基于本数据集 100 秒回测调优,实盘需 walk-forward 验证)
ENTRY_QUANTILE = 0.80     # 信号分位阈值
EXIT_HOLD_SEC = 30        # 最大持仓秒数
TP_BPS = 12.0             # 止盈 bps
SL_BPS = 8.0              # 止损 bps
COOLDOWN_SEC = 3          # 平仓后冷却
MAX_SPREAD_USD = 2.0      # 价差超此值不交易
RISK_PER_TRADE = 0.005    # 单笔风险 0.5%
ACCOUNT_USD = 10_000.0    # 初始账户


def build_signal(df: pd.DataFrame) -> pd.Series:
    """构造复合信号 score, 范围约 [-3, +3]."""
    def z(s: pd.Series) -> pd.Series:
        s = s.fillna(s.median())
        sd = s.std()
        if sd > 0:
            return (s - s.mean()) / sd
        else:
            return pd.Series(0.0, index=s.index)

    score = (
        1.0 * z(df["imbalance_5"]) +
        0.8 * z(df["kline_taker_buy_ratio"] - 0.5) +
        0.6 * z(df["depth_imbalance_02"]) +
        0.5 * z(df["trade_intensity"]) +
        0.4 * z(df["volume_imbalance"])
    )
    return score


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    side: str            # "long" / "short"
    entry_px: float
    exit_px: float
    exit_reason: str
    gross_bps: float
    fees_bps: float
    net_bps: float
    qty: float
    pnl_usd: float


def simulate(df: pd.DataFrame) -> tuple[list[Trade], dict[str, any]]:
    """
    Maker-only 回测:
      - 多头: 在 best_bid 挂买单, 价格跌破 bid 才成交 (真实 maker 逻辑)
      - 空头: 在 best_ask 挂卖单, 价格突破 ask 才成交
      - 平仓: 反向 maker 挂单, 同样需要价格触及
      - 若超过 EXIT_HOLD_SEC 仍未成交平仓, 强制吃单 (taker fee) 平仓
    """
    df = df.reset_index(drop=True).copy()
    df["score"] = build_signal(df)
    df["spread_usd"] = df["best_ask_price"] - df["best_bid_price"]

    # 使用 expanding window 计算分位数避免前视偏差
    long_th = df["score"].expanding(min_periods=20).quantile(ENTRY_QUANTILE)
    short_th = df["score"].expanding(min_periods=20).quantile(1 - ENTRY_QUANTILE)

    trades: list[Trade] = []
    pos = None  # dict: side, entry_idx, entry_px, tp_px, sl_px, qty
    cooldown_until = -1

    for i in range(len(df) - 1):
        row = df.iloc[i]

        # ---- 持仓中: 处理平仓 ----
        if pos is not None:
            held = i - pos["entry_idx"]
            nxt = df.iloc[i + 1]
            hi, lo = nxt["high"], nxt["low"]

            exit_reason = None
            exit_px = None
            exit_is_taker = False

            if pos["side"] == "long":
                # 止损先于止盈触发更保守, 加入滑点
                if lo <= pos["sl_px"]:
                    exit_reason = "stop_loss"
                    exit_px = pos["sl_px"] * 0.9999  # 止损滑点
                    exit_is_taker = True  # 止损必须吃单
                elif hi >= pos["tp_px"]:
                    exit_reason = "take_profit"
                    exit_px = pos["tp_px"]   # maker 挂卖在 tp_px, 价格触及成交
                    exit_is_taker = False
                elif held >= EXIT_HOLD_SEC:
                    exit_reason = "timeout"
                    exit_px = nxt["close"]
                    exit_is_taker = True
            else:  # short
                if hi >= pos["sl_px"]:
                    exit_reason = "stop_loss"
                    exit_px = pos["sl_px"] * 1.0001  # 止损滑点
                    exit_is_taker = True
                elif lo <= pos["tp_px"]:
                    exit_reason = "take_profit"
                    exit_px = pos["tp_px"]
                    exit_is_taker = False
                elif held >= EXIT_HOLD_SEC:
                    exit_reason = "timeout"
                    exit_px = nxt["close"]
                    exit_is_taker = True

            if exit_reason is not None:
                if pos["side"] == "long":
                    gross = (exit_px - pos["entry_px"]) / pos["entry_px"] * 10000
                else:
                    gross = (pos["entry_px"] - exit_px) / pos["entry_px"] * 10000
                entry_fee = MAKER_FEE_BPS  # 入场是 maker
                exit_fee = TAKER_FEE_BPS if exit_is_taker else MAKER_FEE_BPS
                net = gross - entry_fee - exit_fee
                pnl = pos["qty"] * pos["entry_px"] * net / 10000
                trades.append(Trade(
                    entry_idx=pos["entry_idx"], exit_idx=i + 1,
                    side=pos["side"], entry_px=pos["entry_px"],
                    exit_px=exit_px, exit_reason=exit_reason,
                    gross_bps=gross, fees_bps=entry_fee + exit_fee,
                    net_bps=net, qty=pos["qty"], pnl_usd=pnl,
                ))
                pos = None
                cooldown_until = i + COOLDOWN_SEC
                continue

        # ---- 空仓: 处理开仓 ----
        if pos is None and i >= cooldown_until:
            if row["spread_usd"] > MAX_SPREAD_USD:
                continue
            nxt = df.iloc[i + 1]
            score = row["score"]

            # Long: 在 best_bid 挂买单, 价格跌破才成交 (真实 maker)
            if score >= long_th.iloc[i]:
                bid_px = row["best_bid_price"]
                # Maker 逻辑: 价格必须跌破 bid 才能成交
                if nxt["low"] < bid_px:
                    # 单笔风险 = qty * entry_px * SL_BPS/10000 = ACCOUNT * RISK
                    qty = ACCOUNT_USD * RISK_PER_TRADE / (bid_px * SL_BPS / 10000)
                    pos = {
                        "side": "long", "entry_idx": i + 1, "entry_px": bid_px,
                        "tp_px": bid_px * (1 + TP_BPS / 10000),
                        "sl_px": bid_px * (1 - SL_BPS / 10000),
                        "qty": qty,
                    }
            elif score <= short_th.iloc[i]:
                ask_px = row["best_ask_price"]
                # Maker 逻辑: 价格必须突破 ask 才能成交
                if nxt["high"] > ask_px:
                    qty = ACCOUNT_USD * RISK_PER_TRADE / (ask_px * SL_BPS / 10000)
                    pos = {
                        "side": "short", "entry_idx": i + 1, "entry_px": ask_px,
                        "tp_px": ask_px * (1 - TP_BPS / 10000),
                        "sl_px": ask_px * (1 + SL_BPS / 10000),
                        "qty": qty,
                    }

    # 收盘强平
    if pos is not None:
        last = df.iloc[-1]
        exit_px = last["close"]
        if pos["side"] == "long":
            gross = (exit_px - pos["entry_px"]) / pos["entry_px"] * 10000
        else:
            gross = (pos["entry_px"] - exit_px) / pos["entry_px"] * 10000
        net = gross - MAKER_FEE_BPS - TAKER_FEE_BPS
        pnl = pos["qty"] * pos["entry_px"] * net / 10000
        trades.append(Trade(
            entry_idx=pos["entry_idx"], exit_idx=len(df) - 1,
            side=pos["side"], entry_px=pos["entry_px"], exit_px=exit_px,
            exit_reason="eod_close", gross_bps=gross,
            fees_bps=MAKER_FEE_BPS + TAKER_FEE_BPS,
            net_bps=net, qty=pos["qty"], pnl_usd=pnl,
        ))

    metrics = compute_metrics(trades)
    return trades, metrics


def compute_metrics(trades: list[Trade]) -> dict[str, any]:
    if not trades:
        return {"n_trades": 0}
    nets = np.array([t.net_bps for t in trades])
    pnls = np.array([t.pnl_usd for t in trades])
    wins = nets > 0
    return {
        "n_trades": len(trades),
        "n_wins": int(wins.sum()),
        "n_losses": int((~wins).sum()),
        "win_rate": float(wins.mean()),
        "avg_net_bps": float(nets.mean()),
        "median_net_bps": float(np.median(nets)),
        "total_pnl_usd": float(pnls.sum()),
        "total_pnl_pct": float(pnls.sum() / ACCOUNT_USD * 100),
        "max_win_bps": float(nets.max()),
        "max_loss_bps": float(nets.min()),
        "avg_fees_bps": float(np.mean([t.fees_bps for t in trades])),
        "exit_breakdown": {
            r: int(sum(1 for t in trades if t.exit_reason == r))
            for r in {t.exit_reason for t in trades}
        },
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/aggregated/realtime_btcusdt_20260514_112513.csv"
    df = pd.read_csv(path)
    trades, m = simulate(df)
    print(f"\n=== 回测结果 [{path}] ===")
    print(f"数据点: {len(df)}  时间: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    print(f"成交笔数: {m['n_trades']}  胜: {m.get('n_wins',0)}  负: {m.get('n_losses',0)}  胜率: {m.get('win_rate',0):.1%}")
    print(f"平均净收益: {m.get('avg_net_bps',0):+.2f} bps   中位数: {m.get('median_net_bps',0):+.2f} bps")
    print(f"最大单笔盈/亏: {m.get('max_win_bps',0):+.2f} / {m.get('max_loss_bps',0):+.2f} bps")
    print(f"总 PnL: ${m.get('total_pnl_usd',0):+.2f}  ({m.get('total_pnl_pct',0):+.3f}% of ${ACCOUNT_USD:.0f})")
    print(f"平均手续费: {m.get('avg_fees_bps',0):.2f} bps/笔")
    print(f"平仓原因分布: {m.get('exit_breakdown',{})}")
    if trades:
        print("\n--- 前 10 笔交易明细 ---")
        for t in trades[:10]:
            print(f"  #{t.entry_idx:3d}→{t.exit_idx:3d} {t.side:5s} @{t.entry_px:.2f}→{t.exit_px:.2f}  "
                  f"gross={t.gross_bps:+6.2f}bps fees={t.fees_bps:.1f}bps net={t.net_bps:+6.2f}bps "
                  f"pnl=${t.pnl_usd:+.2f}  [{t.exit_reason}]")
