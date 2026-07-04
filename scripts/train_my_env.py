# -*- coding: utf-8 -*-
"""
用你自己写的环境训练PPO
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from envs.Mso_arm101_reach import SoArm101ReachEnv


def make_env():
    return SoArm101ReachEnv(render_mode=None)


def main():
    print("=" * 60)
    print("Training PPO with YOUR Environment")
    print("=" * 60)

    # 1. 创建并行环境
    print("\n1. Creating environments...")
    train_env = SubprocVecEnv([make_env for _ in range(4)])
    eval_env = SubprocVecEnv([make_env])

    # 2. 创建PPO模型
    print("2. Creating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./logs/my_env/",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128])
        ),
    )

    # 3. 评估回调
    print("3. Setting up evaluation...")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/my_env/",
        log_path="./logs/my_env/eval/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
    )

    # 4. 训练
    print("4. Starting training...")
    print("   Total timesteps: 200,000")
    print("   Monitor: tensorboard --logdir=./logs/my_env/")
    print()

    model.learn(
        total_timesteps=5000000,
        callback=eval_callback,
    )

    # 5. 保存
    print("\n5. Saving model...")
    model.save("ppo_my_env_reach")
    print("   Saved to: ppo_my_env_reach.zip")

    train_env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
