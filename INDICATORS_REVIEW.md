# 指标生成代码审查报告

审查范围：`src/aggregators/second_aggregator.py`（实时聚合）、`src/collectors/history_collector.py`（历史聚合）、`main.py:save_aggregated_data`（落盘补齐）。
审查方式：源码逐行通读 + 用 `BTCUSDT_2026-05-12_2026-05-13.csv` 前 10000 条做指标一致性回归。
审查日期：2026-05-14（同日已应用下文所有 P0/P1/P2/P3 修复）。

---

## 0. 修复后的端到端回归（2026-05-12 全天）

| 维度 | 结果 |
|---|---|
| 时间戳严格连续（86399 条 + 补齐 4233 秒） | ✅ |
| OHLC / VWAP / volume 拆分等恒等关系 | ✅ |
| `trade_flow_toxicity` 字段已下线 | ✅ |
| 新增 `depth_age_seconds`（快照陈旧度）| ✅ 中位 0s / p99 ≈ 30s / max 280s |
| 大单检测真实命中（此前被全局 p99.9 过滤砍掉） | ✅ 46% 秒有大单，max_trade_size p99 = 2.7 BTC |
| 聚合性能（270 万 trades + 32k 深度快照 / 天） | ✅ 16.8s（旧 groupby+iterrows 约慢数十倍）|

---

## 1. 总览结论

| 维度 | 结论 |
|---|---|
| OHLC 内部一致性（high≥open/close, low≤open/close） | ✅ 通过 |
| VWAP ∈ [low, high] | ✅ 通过 |
| `buy_volume + sell_volume == volume` | ✅ 通过（浮点误差 < 1e-13） |
| `buy_count + sell_count == trade_count` | ✅ 严格相等 |
| `up_tick_count + down_tick_count == tick_count` | ✅ 严格相等 |
| `trade_flow_toxicity == |volume_imbalance|` | ✅ 完全相等 |
| 时间戳秒级严格连续（落盘后） | ✅ 已修复并验证 |
| 实时与历史路径指标定义一致 | ⚠️ 部分不一致，见 §3.1 |
| 大单检测在小样本下的稳定性 | ⚠️ 见 §3.2 |
| 订单簿对齐方式 | ⚠️ 见 §3.3 |
| 异常值过滤的副作用 | ⚠️ 见 §3.4 |

整体可用，无致命错误。下文按"严重 → 一般 → 风格"排序列出问题与建议。

---

## 2. 指标定义清单（实测语义）

| 指标 | 定义 | 单位 | 取值域 | 来源 |
|---|---|---|---|---|
| `open / high / low / close` | 该秒内成交价的首/最大/最小/末值 | USDT | > 0 | trades |
| `volume` | 该秒成交量之和 | base | ≥ 0 | trades |
| `quote_volume` | Σ price·qty | quote | ≥ 0 | trades |
| `vwap` | quote_volume / volume | USDT | ∈ [low, high] | trades |
| `trade_count` | 该秒成交笔数 | int | ≥ 0 | trades |
| `buy_count / sell_count` | `is_buyer_maker=False/True` 拆分 | int | ≥ 0 | trades |
| `buy_volume / sell_volume` | 同上分组的 qty 和 | base | ≥ 0 | trades |
| `trade_intensity` | float(trade_count) | — | ≥ 0 | trades |
| `avg_trade_size` | volume / trade_count | base | > 0 | trades |
| `max_trade_size` | qty 最大值 | base | > 0 | trades |
| `price_range` | high − low | USDT | ≥ 0 | trades |
| `tick_count` | 价格变动笔数（diff ≠ 0） | int | ≥ 0 | trades |
| `up_tick_count / down_tick_count` | 价格上/下移笔数 | int | ≥ 0 | trades |
| `volume_imbalance` | (buy − sell) / volume | float | ∈ [−1, 1] | trades |
| `trade_flow_toxicity` | abs(buy − sell) / volume | float | ∈ [0, 1] | trades |
| `large_trade_count / volume` | qty > mean+2·std 的笔数与量 | int / base | ≥ 0 | trades |
| `bid_depth_02 / ask_depth_02` | ±0.2% 档位深度 | base | ≥ 0 | bookDepth |
| `bid_depth_1 / ask_depth_1` | ±1% 档位深度 | base | ≥ 0 | bookDepth |
| `total_bid_depth / total_ask_depth` | 全档位深度合计 | base | ≥ 0 | bookDepth |
| `depth_imbalance_02 / 1 / total` | (bid − ask) / (bid + ask) | float | ∈ [−1, 1] | bookDepth |
| `kline_*` | 分钟 K 线特征（见 `_compute_kline_features`） | 多 | — | klines |
| `kline_taker_buy_ratio` | taker_buy_volume / volume | float | ∈ [0, 1] | klines |
| `kline_body_ratio` | \|close−open\| / (high−low) | float | ∈ [0, 1] | klines |
| `kline_upper/lower_shadow` | 上/下影线占总幅度比例 | float | ∈ [0, 1] | klines |
| `update_count` | 该秒订单簿更新次数（实时模式） | int | ≥ 0 | depth ws |
| `best_bid/ask_price/qty / spread_bps / mid_price / imbalance_5` | 实时模式订单簿快照特征 | 多 | — | depth ws |

实测验证（`BTCUSDT_2026-05-12_2026-05-13.csv`, 10000 行）：所有可量化的恒等关系全部成立。

---

## 3. 发现的问题

### 3.1 实时与历史聚合的指标定义不完全一致 ⚠️

`SecondAggregator._compute_orderbook_features`（实时）输出 `best_bid_price / best_ask_price / spread_bps / mid_price / imbalance_5 / update_count`，
而 `HistoryCollector` 输出 `bid_depth_02/1/total / depth_imbalance_02/1/total / total_bid/ask_depth`。

两路的订单簿字段几乎不重叠。这意味着：
- 用历史数据训练、用实时数据推理的模型，特征列对不上。
- `dropna(axis=1, how='all')` 在 `save_aggregated_data` 中会"按数据集自动剔除"全空列，导致两批数据列结构不同（一份没有 `kline_*`、另一份没有 `best_bid_*`），下游若按列名拼接会报错。

**建议**：
- 短期：在 README 或本文件中明确两条路径输出列差异，下游按"指标族"而非"列"对齐。
- 中期：让两条路径都输出对方的字段（实时路径根据 best bid/ask 推算 ±0.2% 深度；历史路径从 bookDepth 取 best_bid/ask 近似）。或者引入显式的 `mode` 字段并把缺失列填 NaN，让下游自行决定如何用。

### 3.2 大单检测阈值在小样本上极易误判 ⚠️

历史路径 `history_collector.py:343-350` 与实时路径 `second_aggregator.py:339-348`：

```python
if len(qtys) > 1:
    threshold = mean + 2 * std
    large_count = (qtys > threshold).sum()
```

问题：
- **样本极小（2~5 笔）时 std 极不稳定**，一个偶发大单会同时拉高 mean 和 std，反而让自身检测不到。低活跃秒（绝大多数秒只有几笔）下该指标信噪比很低。
- **`numpy.std` 默认 `ddof=0`，pandas Decimal 路径用手算 ddof=0**，两路一致；但严格统计意义上小样本应 `ddof=1`，影响虽小、口径需统一记录。
- 当 `trade_count == 1` 时该指标恒为 0（实测确认），下游若把"无大单"和"未定义"区分，需要单独看 `trade_count`。

**建议**：
- 改为按"全局/滚动窗口"阈值（例如全数据集 99 分位数，或近 N 秒滚动均值的 K 倍）。这与 `history_collector.load_trades` 已有的 `qty.quantile(0.999)` 过滤思路一致。
- 或者把"窗口内大单"留给下游计算，本层只输出 `qty` 分布的高阶矩（mean / std / max / p99），让特征更稳定。

### 3.3 订单簿对齐为 LOCF（Last Observation Carry Forward）⚠️

`history_collector.aggregate_to_seconds:281` 用 `mask = depth_features.index <= ts; depth_features.loc[mask].iloc[-1]` 取最近一次快照，再在 `save_aggregated_data` 里 `ffill().bfill()`。

注意点：
- bookDepth 是 ~30s 一拍，碰到剧烈行情时本秒的"深度"实际是几十秒前的视图，**特征滞后**。
- `bfill` 只对开头几秒的 NaN 起作用，但首批 bfill 用的是"未来"快照——回测里这是**未来函数泄露**（look-ahead bias）。

**建议**：
- 移除 `bfill`，让数据集开头的 NaN 暴露给下游（下游可选择丢弃首段或填 0）。
- 保留 `ffill`（合理），但增设 `depth_age_seconds` 字段记录"当前快照距今多少秒"，方便下游识别失真。

### 3.4 异常值过滤会改写聚合输入 ⚠️

`history_collector.load_trades:99-105`：
- 滚动中位数 ±3% 视为异常并丢弃
- qty > 全局 99.9 分位数 丢弃

后果：
- **`large_trade_*` 永远看不到真正的尾部大单**——全局 p99.9 已经被砍掉。和"大单检测"目标自相矛盾。
- 滚动 3% 阈值在闪崩/插针场景会把真实成交也当成异常。BTC 1 秒内 3% 罕见但存在。

**建议**：
- 把"清洗"和"原始聚合"分两层。先按原始 trades 聚合、再在数据集层面做 winsorize/outlier flagging，并保留 `is_outlier_filtered` 字段供回溯。
- 至少把 qty 上界改为 p99.99 或 IQR 法，避免抹掉合法大单。

### 3.5 `_compute_depth_features` 用 `groupby + iterrows` 性能瓶颈

`history_collector.py:225-262` 在每个时间戳分组后再 `iterrows` 拼字段。bookDepth 一天约 2880 个快照×10 档 = 28800 行，单天还能跑，多日并行或大批量回测会很慢。

**建议**：换 `pivot_table(index='timestamp', columns='percentage', values='depth')` 做向量化，再按列重命名/算 imbalance。预计快 10×~50×。

### 3.6 vwap 在 `volume == 0` 时的取值

历史路径 `history_collector.py:310`：`vwap = quote_qtys.sum() / qtys.sum() if qtys.sum() > 0 else Decimal('0')`
实时路径 `second_aggregator.py:309`：`vwap = quote_volume / volume if volume > 0 else Decimal('0')`

`vwap = 0` 在 BTC 数据集里就是异常值，且和"无成交"语义混淆。补齐逻辑（`save_aggregated_data`）已经把缺失秒的 vwap 用上一秒 close 替换，所以 CSV 里看不到 0；但若下游绕过补齐直接消费 `UnifiedMarketData`，会拿到 vwap=0。

**建议**：在 dataclass 层就把 `volume == 0` 时的 vwap 写成 `None`，让缺失语义清晰。

### 3.7 `tick_count` 与 `update_count` 命名混用

- `tick_count` = 成交价变动笔数（trades 域）
- `update_count` = 订单簿更新次数（orderbook 域）

两者都叫 "tick/update count" 但来源不同。文档里需点名。当前命名没错但容易误读。

### 3.8 `trade_flow_toxicity` 与 `volume_imbalance` 信息冗余

实测 `toxicity == |imbalance|`。VPIN 的本意是滚动窗口的累计净买卖比，单秒做绝对值并不能反映"流动性提供者面对的逆向选择风险"。

**建议**：要么删掉 `trade_flow_toxicity`、要么改为 N 秒/N 笔窗口聚合后再计算（典型 50~500 笔窗口）。

### 3.9 历史模式按天调用 → 跨天日界缺失

`history_collector.collect_day` 每天独立处理。一天的最后一秒和次日第一秒之间的 `tick_count`、价格 diff 在 `aggregate_to_seconds` 内没有跨天累计。当前实现里 `up_tick_count` 是单秒内的，不受影响；但若以后加 N 秒滚动特征要注意拼接时的边界。

### 3.10 `_aggregation_loop` 在水位线推进时丢弃空秒 ⚠️

`second_aggregator.py:160-165`：

```python
if second in self.orderbook_buffer or second in self.trade_buffer:
    await self._aggregate_second(second)
```

实时路径中"该秒无任何事件"则不会调用 `_aggregate_second`，导致下游 `on_aggregated_data` 收不到这一秒——和历史路径同样的"零成交秒缺失"问题。`save_aggregated_data` 现在能补上，但**实时单次落盘 `_save_batch`（每 100 条）在跨缺失秒时无法补齐**（缺时间戳引用），下游若直接消费回调流仍会拿到不连续序列。

**建议**：实时路径里，每秒水位线推进时都构造一条 `UnifiedMarketData`，没数据就用上一秒 close + 0 量填充。这样在 dataclass 流上就严格连续。

---

## 4. 风格与小问题

- `history_collector.py:44-45` 字符串里带了无意义换行：`requests.get(url, stream=\n                            True, ...)`，只是难看。
- `second_aggregator.py:341` 用 `Decimal('0.5')` 作 std 的指数：可读，但慢于 numpy 路径；考虑统一改 numpy。
- `_compute_features` 多次 `Decimal(str(x))` 转换，开销集中在历史回放时。批量场景可保留 numpy 直到最后落盘再转。
- `history_collector.py:280-289`：`mask = depth_features.index <= ts; depth_features.loc[mask].iloc[-1]` 在每秒都重扫一遍 depth_features，O(N·M)。换 `merge_asof` 或 `searchsorted` 后是 O(N+M)。

---

## 5. 修复优先级建议

| 优先级 | 问题 | 动作 |
|---|---|---|
| P0 | §3.10 实时路径空秒丢失 | 在 `_aggregation_loop` 里强制每秒产出一条记录 |
| P0 | §3.4 大单与全局过滤冲突 | 大单检测使用未过滤的原始数据 |
| P1 | §3.3 bookDepth 的 bfill 引入 look-ahead | 移除 `bfill`，加 `depth_age_seconds` |
| P1 | §3.1 实时/历史列差异 | 统一 schema 或文档化 |
| P2 | §3.5 depth groupby 性能 | 改 `pivot_table` |
| P2 | §3.2 大单阈值小样本 | 改全局/滚动阈值 |
| P3 | §3.6 vwap=0 语义 | 改为 None |
| P3 | §3.8 toxicity 冗余 | 删除或改 VPIN 实现 |

---

## 6. 已通过的回归（10000 行采样）

- `(high ≥ open, close) ∧ (low ≤ open, close)`：100%
- `low·0.9999 ≤ vwap ≤ high·1.0001`（有成交秒）：100%
- `|buy_volume + sell_volume − volume|` max：5.7e-14（浮点容差内）
- `buy_count + sell_count == trade_count`：100% 严格相等
- `up_tick_count + down_tick_count == tick_count`：100% 严格相等
- `trade_flow_toxicity == |volume_imbalance|`：完全一致
- `depth_imbalance_total ∈ [−0.18, 0.18]`，符合订单簿不平衡的合理范围
- `bid_depth_02` 无 NaN（ffill+bfill 已生效）
- `kline_taker_buy_ratio ∈ [0.05, 0.88]`，无越界
- `kline_body_ratio ∈ [0.005, 1.0]`，无越界
- 单笔成交秒（`trade_count == 1`）的 `large_trade_count` 全为 0（与代码一致）

---

## 7. 一句话总结

代码功能正确、字段口径自洽，但有两处需要尽快修：**实时路径不补缺失秒**（与历史路径行为不一致）和**大单检测被全局 qty 过滤砍掉**（自相矛盾）。其余多为信噪比与性能优化，可按 §5 优先级渐进处理。
