# -*- coding: utf-8 -*-
"""
训练SO-ARM101到达任务
使用PPO算法
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np


def make_env(isrender = None):
    """创建环境的工厂函数"""
    from envs.so_arm101_reach import SOArm101ReachEnv
    return SOArm101ReachEnv(render_mode=isrender)


def main():
    print("=" * 60)
    print("SO-ARM101 Reach Task Training")
    print("=" * 60)

    # 1. 创建向量化环境（4个并行环境加速训练）
    print("\n1. Creating environments...")
    train_env = SubprocVecEnv([make_env for _ in range(4)])
    eval_env = SubprocVecEnv([make_env])

    # 2. 创建PPO模型
    print("2. Creating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,      # 学习率
        n_steps=2048,             # 每次更新的步数
        batch_size=64,            # 批次大小
        n_epochs=10,              # 每次更新的epoch数
        gamma=0.99,               # 折扣因子（重视未来奖励）
        gae_lambda=0.95,          # GAE lambda
        clip_range=0.2,           # PPO clipping
        verbose=1,                # 打印训练信息
        tensorboard_log="./logs/reach_task/",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128])  # 网络结构
        ),
    )

    # 3. 设置评估回调
    print("3. Setting up evaluation callback...")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/",
        log_path="./logs/eval/",
        eval_freq=10000,          # 每10000步评估一次
        n_eval_episodes=5,        # 每次评估5个episode
        deterministic=True,       # 评估时用确定性策略
        render=False,
    )

    # 4. 开始训练
    print("4. Starting training...")
    print("   Total timesteps: 200,000")
    print("   You can monitor with: tensorboard --logdir=./logs/")
    print()

    model.learn(
        total_timesteps=2000000,
        callback=eval_callback,
    )

    # 5. 保存最终模型
    print("\n5. Saving final model...")
    model.save("ppo_so_arm101_reach")
    print("   Model saved to: ppo_so_arm101_reach.zip")

    # 6. 快速测试
    print("\n6. Quick test...")
    test_env = make_env()
    obs, _ = test_env.reset()

    total_reward = 0
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    ee_pos = obs[6:9]
    target_pos = obs[9:12]
    distance = np.linalg.norm(ee_pos - target_pos)
    print(f"   Final distance: {distance:.4f} m")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Steps: {step + 1}")
    print(f"   Success: {distance < 0.02}")

    test_env.close()
    train_env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

def test_reach():
    print("\n6. Quick test...")
    test_env = make_env("human")
    obs, _ = test_env.reset()
    model = PPO.load("./checkpoints/best_model.zip")
    total_reward = 0
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    ee_pos = obs[6:9]
    target_pos = obs[9:12]
    distance = np.linalg.norm(ee_pos - target_pos)
    print(f"   Final distance: {distance:.4f} m")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Steps: {step + 1}")
    print(f"   Success: {distance < 0.02}")

    test_env.close()

if __name__ == "__main__":
    # main()
    test_reach()
