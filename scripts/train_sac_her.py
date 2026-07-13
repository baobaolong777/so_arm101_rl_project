# -*- coding: utf-8 -*-
"""
SAC + HER 训练 reach 任务
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.her import HerReplayBuffer
from envs.Mso_arm101_reach import SoArm101ReachEnv


class ReachDictWrapper(gym.ObservationWrapper):
    """将 flat observation 转为 dict observation，适配 HER"""
    def __init__(self, env):
        super().__init__(env)
        # 原始 obs: [joint_angles(6), joint_velocities(6), ee_pos(3), target_pos(3)] = 18
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),  # joint_angles + joint_velocities
            "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),  # ee_pos
            "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),   # target_pos
        })

    def observation(self, obs):
        return {
            "observation": obs[:12].astype(np.float32),    # joint_angles + joint_velocities
            "achieved_goal": obs[12:15].astype(np.float32), # ee_pos
            "desired_goal": obs[15:18].astype(np.float32),  # target_pos
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        """HER 需要的奖励计算：基于 achieved_goal 和 desired_goal"""
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(distance > 0.02).astype(np.float32)  # 稀疏奖励：0.02m内=0，否则=-1


def make_env():
    env = SoArm101ReachEnv(render_mode=None)
    env = ReachDictWrapper(env)
    return env


def main():
    print("=" * 60)
    print("Training SAC + HER for Reach Task")
    print("=" * 60)

    # 1. 创建环境
    print("\n1. Creating environments...")
    train_env = DummyVecEnv([make_env for _ in range(8)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 2. 创建 SAC + HER 模型
    print("2. Creating SAC + HER model...")
    model = SAC(
        policy="MultiInputPolicy",
        env=train_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.95,
        tau=0.05,
        ent_coef="auto",
        learning_starts=10000,
        verbose=1,
        tensorboard_log="./logs/my_env/",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], qf=[256, 256])
        ),
    )

    # 3. 评估回调
    print("3. Setting up evaluation...")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/sac_her/",
        log_path="./logs/my_env/sac_her_eval/",
        eval_freq=10000,
        n_eval_episodes=20,
        deterministic=True,
    )

    # 4. 训练
    print("4. Starting training...")
    print("   Total timesteps: 500,000")
    print("   Monitor: tensorboard --logdir=./logs/my_env/")
    print()

    model.learn(
        total_timesteps=5000000,
        callback=eval_callback,
    )

    # 5. 保存
    print("\n5. Saving model...")
    model.save("models/sac_her_reach")
    train_env.save("vec_normalize/vec_normalize_sac.pkl")
    print("   Saved to: models/sac_her_reach.zip + vec_normalize/vec_normalize_sac.pkl")

    train_env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
