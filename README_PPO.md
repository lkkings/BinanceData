# PPO-Based Market Maker 在线强化学习系统

基于Ray RLlib的PPO算法实现的做市商强化学习系统，支持专家数据、评估和断点续训。

## 功能特性

- ✅ **PPO算法**: 使用Ray RLlib 2.x实现的Proximal Policy Optimization
- ✅ **专家数据支持**: 可以使用专家演示数据进行模仿学习
- ✅ **自动评估**: 每轮训练后自动进行评估
- ✅ **断点续训**: 支持保存和恢复训练检查点
- ✅ **Binance数据集成**: 与Binance数据采集系统无缝集成

## 系统架构

```
test.py
├── MarketMakerEnv          # Gym环境实现
├── MarketMakerCallbacks    # 自定义评估回调
├── CheckpointManager       # 检查点管理
├── ExpertDataHandler       # 专家数据处理
├── create_ppo_config()     # PPO配置
└── 训练函数
    ├── train_from_scratch()      # 从头训练
    ├── continue_training()       # 继续训练
    ├── train_with_expert_data()  # 使用专家数据训练
    └── evaluate_model()          # 模型评估
```

## 环境说明

### 观察空间 (11维)
- mid_price: 中间价 (归一化)
- spread_bps: 价差(基点)
- bid_depth_5: 买单深度(前5档)
- ask_depth_5: 卖单深度(前5档)
- imbalance_5: 订单簿不平衡度
- vwap: 成交量加权平均价
- volume: 成交量
- buy_sell_ratio: 买卖比率
- volatility: 波动率
- inventory: 当前持仓
- total_pnl: 总盈亏

### 动作空间 (2维连续)
- action[0]: 买单偏移 (相对于价差的比例, [-1, 1])
- action[1]: 卖单偏移 (相对于价差的比例, [-1, 1])

### 奖励函数
```python
reward = spread_reward + fill_reward - inventory_penalty
```
- spread_reward: 报价价差奖励
- fill_reward: 成交奖励
- inventory_penalty: 持仓惩罚

## 安装依赖

```bash
# 安装Ray RLlib和相关依赖
pip install ray[rllib] torch gymnasium pandas numpy

# 或使用项目的pyproject.toml
uv sync
```

## 使用方法

### 1. 从头开始训练

```bash
python test.py --mode train --iterations 100 --checkpoint_freq 10
```

参数说明:
- `--iterations`: 训练迭代次数 (默认: 100)
- `--checkpoint_freq`: 检查点保存频率 (默认: 10)

### 2. 继续训练

```bash
# 从最新检查点继续
python test.py --mode continue --iterations 50

# 从指定检查点继续
python test.py --mode continue --checkpoint ./checkpoints/checkpoint_000050 --iterations 50
```

### 3. 使用专家数据训练

```bash
# 使用合成专家数据
python test.py --mode train_expert --iterations 100

# 使用自定义专家数据
python test.py --mode train_expert --expert_data ./my_expert_data.json --iterations 100
```

### 4. 评估模型

```bash
# 评估最佳模型
python test.py --mode evaluate --num_episodes 10

# 评估指定检查点
python test.py --mode evaluate --checkpoint ./checkpoints/checkpoint_000100 --num_episodes 10
```

## 数据准备

系统需要聚合后的市场数据，数据格式应包含以下字段:

```csv
timestamp,symbol,mid_price,spread_bps,bid_depth_5,ask_depth_5,imbalance_5,vwap,volume,buy_sell_ratio,volatility
```

数据路径配置:
- 默认路径: `./data/aggregated/*.csv`
- 可通过环境配置修改

## 配置参数

### 环境配置
```python
env_config = {
    "max_inventory": 10.0,          # 最大持仓
    "inventory_penalty": 0.01,      # 持仓惩罚系数
    "spread_reward_weight": 1.0,    # 价差奖励权重
    "fill_reward": 0.1,             # 成交奖励
    "tick_size": 0.01,              # 最小价格变动
    "max_steps": 1000,              # 每个episode最大步数
    "data_path": "./data/aggregated"
}
```

### PPO配置
```python
- num_workers: 2                    # 并行worker数量
- train_batch_size: 4000           # 训练批次大小
- sgd_minibatch_size: 128          # SGD小批次大小
- num_sgd_iter: 30                 # SGD迭代次数
- lr: 3e-4                         # 学习率
- gamma: 0.99                      # 折扣因子
- lambda_: 0.95                    # GAE lambda
- clip_param: 0.2                  # PPO裁剪参数
- entropy_coeff: 0.01              # 熵系数
```

## 评估指标

训练过程中会自动记录以下指标:

- **episode_reward_mean**: 平均回合奖励
- **avg_inventory**: 平均持仓量
- **max_inventory**: 最大持仓量
- **final_pnl**: 最终盈亏
- **max_pnl**: 最大盈亏
- **sharpe_ratio**: 夏普比率

## 检查点管理

检查点自动保存在 `./checkpoints/` 目录:

```
checkpoints/
├── checkpoint_000010/
├── checkpoint_000020/
├── ...
└── metadata.json          # 检查点元数据
```

元数据包含:
- 检查点路径
- 训练迭代次数
- 时间戳
- 性能指标

## 专家数据格式

专家数据应为JSON格式:

```json
[
  {
    "obs": [0.5, 0.1, 100.0, ...],
    "actions": [-0.3, 0.3],
    "rewards": 0.5
  },
  ...
]
```

可以使用 `ExpertDataHandler` 生成合成专家数据:

```python
from test import ExpertDataHandler

handler = ExpertDataHandler()
expert_data_path = handler.create_synthetic_expert_data(
    data_path="./data/aggregated",
    strategy="inventory_control"
)
```

## 性能优化建议

1. **并行化**: 增加 `num_workers` 以加速数据采集
2. **批次大小**: 根据GPU内存调整 `train_batch_size`
3. **评估频率**: 降低 `evaluation_interval` 以减少评估开销
4. **检查点频率**: 根据训练时长调整 `checkpoint_freq`

## 故障排查

### 问题: 数据路径不存在
```
FileNotFoundError: Data path ./data/aggregated does not exist
```
**解决**: 确保数据目录存在且包含CSV文件

### 问题: Ray初始化失败
```
RuntimeError: Maybe you called ray.init twice by accident?
```
**解决**: 使用 `ray.init(ignore_reinit_error=True)`

### 问题: GPU内存不足
```
RuntimeError: CUDA out of memory
```
**解决**: 
- 减小 `train_batch_size`
- 设置 `num_gpus=0` 使用CPU训练

## 扩展开发

### 自定义奖励函数

修改 `MarketMakerEnv.step()` 方法中的奖励计算:

```python
def step(self, action):
    # ... 现有代码 ...
    
    # 自定义奖励
    custom_reward = your_reward_function(...)
    reward += custom_reward
    
    return obs, reward, terminated, truncated, info
```

### 添加新的观察特征

1. 修改 `observation_space` 维度
2. 更新 `_get_observation()` 方法
3. 确保数据包含新特征

### 自定义回调

继承 `MarketMakerCallbacks` 并重写方法:

```python
class CustomCallbacks(MarketMakerCallbacks):
    def on_episode_end(self, **kwargs):
        super().on_episode_end(**kwargs)
        # 添加自定义逻辑
```

## 参考文档

- [Ray RLlib Documentation](https://docs.ray.io/en/latest/rllib/index.html)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request!
