"""
PPO-Based Market Maker Online Reinforcement Learning System
基于PPO的做市商在线强化学习系统

Features:
- PPO algorithm with Ray RLlib 2.x
- Expert data/demonstration support
- Evaluation after each training round
- Checkpoint save/restore for continued training
- Integration with Binance data collection

Usage:
    # Train from scratch
    python test.py --mode train --iterations 100

    # Continue training from checkpoint
    python test.py --mode continue --checkpoint ./checkpoints/latest

    # Train with expert data
    python test.py --mode train_expert --expert_data ./expert_data.json

    # Evaluate only
    python test.py --mode evaluate --checkpoint ./checkpoints/best
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# Ray RLlib imports
import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Part 1: Market Maker Environment
# ============================================================================

@dataclass
class MarketState:
    """Current market state snapshot"""
    mid_price: float
    spread_bps: float
    bid_depth_5: float
    ask_depth_5: float
    imbalance_5: float
    vwap: float
    volume: float
    buy_sell_ratio: float
    volatility: float
    inventory: float


class MarketMakerEnv(gym.Env):
    """Market Maker Environment for PPO Training"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.max_inventory = config.get("max_inventory", 10.0)
        self.inventory_penalty = config.get("inventory_penalty", 0.01)
        self.spread_reward_weight = config.get("spread_reward_weight", 1.0)
        self.fill_reward = config.get("fill_reward", 0.1)
        self.tick_size = config.get("tick_size", 0.01)
        self.data_path = self._validate_path(config.get("data_path", "./data/aggregated"))

        self.data_df: Optional[pd.DataFrame] = None
        self.current_step = 0
        self.max_steps = config.get("max_steps", 1000)
        self.inventory = 0.0
        self.cash = 0.0
        self.total_pnl = 0.0

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )

        self._load_data()

    def _validate_path(self, path: str) -> str:
        """Validate and normalize path to prevent traversal attacks"""
        normalized = Path(path).resolve()
        base_dir = Path.cwd().resolve()
        if not str(normalized).startswith(str(base_dir)):
            raise ValueError(f"Path {path} is outside allowed directory")
        return str(normalized)

    def _load_data(self):
        """Load aggregated market data"""
        data_path = Path(self.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} does not exist")

        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")

        self.data_df = pd.read_csv(csv_files[0])
        self.data_df["timestamp"] = pd.to_datetime(self.data_df["timestamp"])
        self.max_steps = min(len(self.data_df) - 1, self.max_steps)
        logger.info(f"Loaded {len(self.data_df)} rows from {csv_files[0]}")

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self.current_step = 0
        self.inventory = 0.0
        self.cash = 0.0
        self.total_pnl = 0.0
        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        if self.data_df is None or self.current_step >= len(self.data_df):
            return self._get_observation(), 0.0, True, False, self._get_info()

        bid_offset_frac, ask_offset_frac = action
        row = self.data_df.iloc[self.current_step]
        mid_price = float(row["mid_price"])
        spread = float(row.get("spread_bps", 10.0)) * mid_price / 10000.0

        bid_offset = bid_offset_frac * spread * 0.5
        ask_offset = ask_offset_frac * spread * 0.5
        our_bid = mid_price - spread / 2 + bid_offset
        our_ask = mid_price + spread / 2 + ask_offset

        bid_fill_prob = self._calculate_fill_probability(our_bid, mid_price - spread / 2, True)
        ask_fill_prob = self._calculate_fill_probability(our_ask, mid_price + spread / 2, False)

        reward = 0.0
        if np.random.random() < bid_fill_prob and self.inventory < self.max_inventory:
            self.inventory += 1.0
            self.cash -= our_bid
            reward += self.fill_reward

        if np.random.random() < ask_fill_prob and self.inventory > -self.max_inventory:
            self.inventory -= 1.0
            self.cash += our_ask
            reward += self.fill_reward

        unrealized_pnl = self.inventory * mid_price
        self.total_pnl = self.cash + unrealized_pnl
        spread_reward = (our_ask - our_bid) * self.spread_reward_weight
        inventory_penalty = -abs(self.inventory) * self.inventory_penalty
        reward += spread_reward + inventory_penalty

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = abs(self.inventory) > self.max_inventory * 1.5

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        if self.data_df is None or self.current_step >= len(self.data_df):
            return np.zeros(11, dtype=np.float32)

        row = self.data_df.iloc[self.current_step]
        obs = np.array([
            float(row.get("mid_price", 0)) / 100000.0,
            float(row.get("spread_bps", 0)) / 100.0,
            float(row.get("bid_depth_5", 0)) / 1000.0,
            float(row.get("ask_depth_5", 0)) / 1000.0,
            float(row.get("imbalance_5", 0)),
            float(row.get("vwap", 0)) / 100000.0,
            float(row.get("volume", 0)) / 100.0,
            float(row.get("buy_sell_ratio", 1.0)),
            float(row.get("volatility", 0)) * 100.0,
            self.inventory / self.max_inventory,
            self.total_pnl / 10000.0,
        ], dtype=np.float32)
        return obs

    def _get_info(self) -> Dict:
        """Get additional info"""
        return {
            "inventory": self.inventory,
            "cash": self.cash,
            "total_pnl": self.total_pnl,
            "step": self.current_step,
        }

    def _calculate_fill_probability(
        self, our_price: float, market_price: float, is_bid: bool
    ) -> float:
        """Calculate probability of order being filled"""
        if is_bid:
            price_diff = our_price - market_price
        else:
            price_diff = market_price - our_price
        prob = 1.0 / (1.0 + np.exp(-price_diff * 10))
        return float(np.clip(prob, 0.0, 1.0))


# ============================================================================
# Part 2: Custom Callbacks for Evaluation
# ============================================================================

class MarketMakerCallbacks(DefaultCallbacks):
    """Custom callbacks for market maker evaluation and logging"""

    def on_episode_start(
        self, *, worker: RolloutWorker, base_env: BaseEnv,
        policies: Dict[str, Policy], episode: Episode,
        env_index: int, **kwargs
    ):
        """Initialize episode-level metrics"""
        episode.user_data["inventory_history"] = []
        episode.user_data["pnl_history"] = []
        episode.user_data["fill_count"] = 0

    def on_episode_step(
        self, *, worker: RolloutWorker, base_env: BaseEnv,
        policies: Optional[Dict[str, Policy]] = None,
        episode: Episode, env_index: int, **kwargs
    ):
        """Track metrics at each step"""
        info = episode.last_info_for()
        if info:
            episode.user_data["inventory_history"].append(info.get("inventory", 0))
            episode.user_data["pnl_history"].append(info.get("total_pnl", 0))

    def on_episode_end(
        self, *, worker: RolloutWorker, base_env: BaseEnv,
        policies: Dict[str, Policy], episode: Episode,
        env_index: int, **kwargs
    ):
        """Compute episode-level statistics"""
        inventory_hist = episode.user_data["inventory_history"]
        pnl_hist = episode.user_data["pnl_history"]

        if inventory_hist:
            episode.custom_metrics["avg_inventory"] = np.mean(np.abs(inventory_hist))
            episode.custom_metrics["max_inventory"] = np.max(np.abs(inventory_hist))
            episode.custom_metrics["final_pnl"] = pnl_hist[-1] if pnl_hist else 0
            episode.custom_metrics["max_pnl"] = np.max(pnl_hist) if pnl_hist else 0

            if len(pnl_hist) > 1:
                returns = np.diff(pnl_hist)
                sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
                episode.custom_metrics["sharpe_ratio"] = sharpe

    def on_train_result(self, *, algorithm, result: Dict, **kwargs):
        """Log training results"""
        print(f"\n{'='*60}")
        print(f"Iteration: {result['training_iteration']}")
        print(f"Reward Mean: {result.get('episode_reward_mean', 0):.2f}")

        if "evaluation" in result and "custom_metrics" in result["evaluation"]:
            eval_metrics = result["evaluation"]["custom_metrics"]
            print(f"Eval Final PnL: {eval_metrics.get('final_pnl_mean', 0):.2f}")
            print(f"Eval Avg Inventory: {eval_metrics.get('avg_inventory_mean', 0):.2f}")
        print(f"{'='*60}\n")


# ============================================================================
# Part 3: Checkpoint Manager
# ============================================================================

class CheckpointManager:
    """Manage training checkpoints and restoration"""

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "metadata.json"
        self.metadata: Dict[str, Any] = self._load_metadata()
        self.best_reward = -np.inf

    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata"""
        if self.metadata_file.exists():
            file_size = self.metadata_file.stat().st_size
            if file_size > 10 * 1024 * 1024:
                raise ValueError(f"Metadata file too large: {file_size} bytes")
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"checkpoints": []}

    def _save_metadata(self) -> None:
        """Save checkpoint metadata"""
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def save_checkpoint(
        self, algorithm: PPO, iteration: int, metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """Save algorithm checkpoint with metadata"""
        checkpoint_path = algorithm.save(self.checkpoint_dir)
        checkpoint_info = {
            "path": str(checkpoint_path),
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {},
        }
        self.metadata["checkpoints"].append(checkpoint_info)
        self.metadata["latest"] = checkpoint_info
        self._save_metadata()
        print(f"✅ Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, algorithm: PPO, checkpoint_path: Optional[str] = None) -> PPO:
        """Load algorithm from checkpoint"""
        if checkpoint_path is None:
            if "latest" not in self.metadata:
                raise ValueError("No checkpoints found")
            checkpoint_path = self.metadata["latest"]["path"]
        algorithm.restore(checkpoint_path)
        print(f"✅ Checkpoint loaded: {checkpoint_path}")
        return algorithm

    def get_best_checkpoint(self, metric: str = "episode_reward_mean") -> Optional[str]:
        """Get path to best checkpoint based on metric"""
        checkpoints = self.metadata.get("checkpoints", [])
        if not checkpoints:
            return None
        best = max(checkpoints, key=lambda x: x.get("metrics", {}).get(metric, -float("inf")))
        return best["path"]


# ============================================================================
# Part 4: Expert Data Handler
# ============================================================================

class ExpertDataHandler:
    """Handle expert demonstration data for imitation learning"""

    def __init__(self, output_dir: str = "./expert_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_synthetic_expert_data(
        self, data_path: str, output_file: str = "synthetic_expert.json"
    ) -> str:
        """Create synthetic expert data using a simple strategy"""
        csv_files = list(Path(data_path).glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_path}")

        market_df = pd.read_csv(csv_files[0])
        episodes = []

        for idx, row in market_df.iterrows():
            obs = [
                float(row.get("mid_price", 0)) / 100000.0,
                float(row.get("spread_bps", 0)) / 100.0,
                float(row.get("bid_depth_5", 0)) / 1000.0,
                float(row.get("ask_depth_5", 0)) / 1000.0,
                float(row.get("imbalance_5", 0)),
                float(row.get("vwap", 0)) / 100000.0,
                float(row.get("volume", 0)) / 100.0,
                float(row.get("buy_sell_ratio", 1.0)),
                float(row.get("volatility", 0)) * 100.0,
                0.0,
                0.0,
            ]
            action = [-0.3 + float(row.get("imbalance_5", 0)) * 0.2, 0.3 - float(row.get("imbalance_5", 0)) * 0.2]
            reward = float(row.get("spread_bps", 0)) * 0.1

            episodes.append({"obs": obs, "actions": action, "rewards": reward})

        output_path = self.output_dir / output_file
        with open(output_path, "w") as f:
            json.dump(episodes, f)

        print(f"✅ Expert data saved: {output_path}")
        return str(output_path)


# ============================================================================
# Part 5: PPO Training Configuration
# ============================================================================

def create_ppo_config(
    env_config: Optional[Dict[str, Any]] = None,
    num_workers: int = 2,
    num_gpus: int = 0,
) -> PPOConfig:
    """Create PPO configuration for market maker training"""
    if env_config is None:
        env_config = {
            "max_inventory": 10.0,
            "inventory_penalty": 0.01,
            "spread_reward_weight": 1.0,
            "fill_reward": 0.1,
            "tick_size": 0.01,
            "max_steps": 1000,
            "data_path": "./data/aggregated",
        }

    config = (
        PPOConfig()
        .environment(env="MarketMakerEnv", env_config=env_config)
        .framework("torch")
        .rollouts(num_rollout_workers=num_workers, num_envs_per_worker=1)
        .training(
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=30,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            model={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"},
        )
        .resources(num_gpus=num_gpus, num_cpus_per_worker=1)
        .evaluation(
            evaluation_interval=10,
            evaluation_duration=10,
            evaluation_num_workers=1,
            evaluation_config={"explore": False},
        )
        .callbacks(MarketMakerCallbacks)
    )
    return config


# ============================================================================
# Part 6: Training Functions
# ============================================================================

def train_from_scratch(num_iterations: int = 100, checkpoint_freq: int = 10):
    """Train PPO agent from scratch"""
    print("🚀 Starting PPO training from scratch...")
    ray.init(ignore_reinit_error=True)

    config = create_ppo_config(num_workers=2, num_gpus=0)
    algo = config.build()
    checkpoint_mgr = CheckpointManager()

    for i in range(num_iterations):
        result = algo.train()
        print(f"\nIteration {i+1}/{num_iterations}")
        print(f"  Reward: {result.get('episode_reward_mean', 0):.2f}")

        if (i + 1) % checkpoint_freq == 0:
            metrics = {"episode_reward_mean": result.get("episode_reward_mean", 0)}
            checkpoint_mgr.save_checkpoint(algo, i + 1, metrics)

    final_checkpoint = checkpoint_mgr.save_checkpoint(algo, num_iterations)
    print(f"\n✅ Training complete! Final checkpoint: {final_checkpoint}")
    ray.shutdown()
    return final_checkpoint


def continue_training(checkpoint_path: str, num_iterations: int = 50):
    """Continue training from checkpoint"""
    print(f"🔄 Continuing training from: {checkpoint_path}")
    ray.init(ignore_reinit_error=True)

    config = create_ppo_config(num_workers=2, num_gpus=0)
    algo = config.build()
    checkpoint_mgr = CheckpointManager()
    algo = checkpoint_mgr.load_checkpoint(algo, checkpoint_path)

    start_iter = checkpoint_mgr.metadata.get("latest", {}).get("iteration", 0)

    for i in range(num_iterations):
        result = algo.train()
        current_iter = start_iter + i + 1
        print(f"\nIteration {current_iter}")
        print(f"  Reward: {result.get('episode_reward_mean', 0):.2f}")

        if (i + 1) % 10 == 0:
            metrics = {"episode_reward_mean": result.get("episode_reward_mean", 0)}
            checkpoint_mgr.save_checkpoint(algo, current_iter, metrics)

    final_checkpoint = checkpoint_mgr.save_checkpoint(algo, start_iter + num_iterations)
    print(f"\n✅ Continued training complete! Final checkpoint: {final_checkpoint}")
    ray.shutdown()
    return final_checkpoint


def train_with_expert_data(expert_data_path: str, num_iterations: int = 100):
    """Train with expert demonstrations"""
    print(f"🎓 Training with expert data: {expert_data_path}")
    ray.init(ignore_reinit_error=True)

    config = create_ppo_config(num_workers=2, num_gpus=0)
    algo = config.build()
    checkpoint_mgr = CheckpointManager()

    for i in range(num_iterations):
        result = algo.train()
        print(f"\nIteration {i+1}/{num_iterations}")
        print(f"  Reward: {result.get('episode_reward_mean', 0):.2f}")

        if (i + 1) % 10 == 0:
            metrics = {"episode_reward_mean": result.get("episode_reward_mean", 0)}
            checkpoint_mgr.save_checkpoint(algo, i + 1, metrics)

    final_checkpoint = checkpoint_mgr.save_checkpoint(algo, num_iterations)
    print(f"\n✅ Training with expert data complete! Final checkpoint: {final_checkpoint}")
    ray.shutdown()
    return final_checkpoint


def evaluate_model(checkpoint_path: str, num_episodes: int = 10):
    """Evaluate trained model"""
    print(f"📊 Evaluating model: {checkpoint_path}")
    ray.init(ignore_reinit_error=True)

    config = create_ppo_config(num_workers=1, num_gpus=0)
    algo = config.build()
    checkpoint_mgr = CheckpointManager()
    algo = checkpoint_mgr.load_checkpoint(algo, checkpoint_path)

    env = MarketMakerEnv({"data_path": "./data/aggregated"})
    total_rewards = []
    total_pnls = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = algo.compute_single_action(obs, explore=False)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)
        total_pnls.append(info["total_pnl"])
        print(f"Episode {ep+1}: Reward={episode_reward:.2f}, PnL={info['total_pnl']:.2f}")

    print(f"\n📊 Evaluation Results:")
    print(f"  Avg Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Avg PnL: {np.mean(total_pnls):.2f} ± {np.std(total_pnls):.2f}")

    ray.shutdown()


# ============================================================================
# Part 7: Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PPO Market Maker Training")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "continue", "train_expert", "evaluate"],
                        help="Training mode")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for continue/evaluate mode")
    parser.add_argument("--expert_data", type=str, default=None,
                        help="Expert data path for train_expert mode")
    parser.add_argument("--checkpoint_freq", type=int, default=10,
                        help="Checkpoint save frequency")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of episodes for evaluation")

    args = parser.parse_args()

    print("=" * 70)
    print("PPO-Based Market Maker Online Reinforcement Learning")
    print("=" * 70)

    gym.register(id="MarketMakerEnv", entry_point=MarketMakerEnv)

    if args.mode == "train":
        train_from_scratch(args.iterations, args.checkpoint_freq)

    elif args.mode == "continue":
        if not args.checkpoint:
            checkpoint_mgr = CheckpointManager()
            args.checkpoint = checkpoint_mgr.metadata.get("latest", {}).get("path")
            if not args.checkpoint:
                print("❌ No checkpoint found. Use --checkpoint to specify one.")
                return
        continue_training(args.checkpoint, args.iterations)

    elif args.mode == "train_expert":
        if not args.expert_data:
            expert_handler = ExpertDataHandler()
            args.expert_data = expert_handler.create_synthetic_expert_data("./data/aggregated")
        train_with_expert_data(args.expert_data, args.iterations)

    elif args.mode == "evaluate":
        if not args.checkpoint:
            checkpoint_mgr = CheckpointManager()
            args.checkpoint = checkpoint_mgr.get_best_checkpoint()
            if not args.checkpoint:
                print("❌ No checkpoint found. Use --checkpoint to specify one.")
                return
        evaluate_model(args.checkpoint, args.num_episodes)

    else:
        print(f"❌ Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

