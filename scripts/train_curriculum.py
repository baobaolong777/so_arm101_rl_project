# -*- coding: utf-8 -*-
"""
两阶段训练：先 0.05m 学会基本 reach，再 0.02m 精细调整
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from envs.Mso_arm101_reach import SoArm101ReachEnv


def make_env(success_dis=0.05):
    def _init():
        env = SoArm101ReachEnv(render_mode=None)
        env.success_dis = success_dis
        return env
    return _init


def main():
    print("=" * 60)
    print("Two-Stage Curriculum Training")
    print("Stage 1: success_dis=0.05m → Stage 2: success_dis=0.02m")
    print("=" * 60)

    # ========== Stage 1: 0.05m ==========
    print("\n>>> Stage 1: Training with success_dis=0.05m ...")
    train_env = DummyVecEnv([make_env(0.05) for _ in range(8)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env(0.05)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

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
        policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128])),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/curriculum/",
        log_path="./logs/my_env/curriculum/",
        eval_freq=50000,
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(total_timesteps=3000000, callback=[eval_callback])

    # 保存 Stage 1 结果
    model.save("models/ppo_stage1_05m")
    train_env.save("vec_normalize/vec_normalize_stage1.pkl")
    train_env.close()
    eval_env.close()
    print("Stage 1 complete!")

    # ========== Stage 2: 0.02m ==========
    print("\n>>> Stage 2: Fine-tuning with success_dis=0.02m ...")
    train_env2 = DummyVecEnv([make_env(0.02) for _ in range(8)])
    train_env2 = VecNormalize(train_env2, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env2 = DummyVecEnv([make_env(0.02)])
    eval_env2 = VecNormalize(eval_env2, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 加载 Stage 1 模型（用新环境继续训练）
    model = PPO.load("models/ppo_stage1_05m", env=train_env2)

    # 复制 Stage 1 的归一化统计量到 Stage 2 环境（深拷贝避免共享引用）
    import copy
    vec_norm_old = VecNormalize.load("vec_normalize/vec_normalize_stage1.pkl", DummyVecEnv([make_env(0.05)]))
    train_env2.obs_rms = copy.deepcopy(vec_norm_old.obs_rms)
    eval_env2.obs_rms = copy.deepcopy(vec_norm_old.obs_rms)
    vec_norm_old.close()

    eval_callback2 = EvalCallback(
        eval_env2,
        best_model_save_path="./checkpoints/curriculum/",
        log_path="./logs/my_env/curriculum/",
        eval_freq=50000,
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(total_timesteps=5000000, callback=[eval_callback2], reset_num_timesteps=False)

    # 保存 Stage 2 结果
    model.save("models/ppo_stage2_02m")
    train_env2.save("vec_normalize/vec_normalize_stage2.pkl")
    train_env2.close()
    eval_env2.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("  Stage 1 model: ppo_stage1_05m.zip")
    print("  Stage 2 model: ppo_stage2_02m.zip")
    print("  Best checkpoint: checkpoints/curriculum/best_model.zip")
    print("=" * 60)


if __name__ == "__main__":
    main()
