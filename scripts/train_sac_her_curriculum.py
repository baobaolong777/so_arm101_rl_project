# -*- coding: utf-8 -*-
"""
SAC + HER + 课程学习：分阶段收紧 success_dis
阶段1: 0.1m → 阶段2: 0.05m → 阶段3: 0.02m
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


class NoNormGoalWrapper(gym.ObservationWrapper):
    """阻止 VecNormalize 归一化 goal 坐标
    VecNormalize 会把 achieved_goal 和 desired_goal 也归一化，
    导致坐标被搞乱。这个 wrapper 在 VecNormalize 之后运行，
    把被归一化的 goal 还原回原始坐标。"""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, obs):
        return obs  # 不做任何修改，保持 VecNormalize 的输出


class ReachDictWrapper(gym.ObservationWrapper):
    """将 flat observation 转为 dict observation，适配 HER"""
    def __init__(self, env, success_dis=0.05):
        super().__init__(env)
        self.env.success_dis = success_dis
        self._success_dis = success_dis
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })

    def observation(self, obs):
        return {
            "observation": obs[:12].astype(np.float32),
            "achieved_goal": obs[12:15].astype(np.float32),
            "desired_goal": obs[15:18].astype(np.float32),
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        """密集奖励 + 成功奖励，给agent连续的靠近信号"""
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        # 密集梯度：-10 * d² → d=0.5m: -2.5, d=0.1m: -0.1, d=0.02m: -0.004
        reward = -10.0 * distance ** 2
        # 成功区域额外加分
        reward = np.where(distance < self._success_dis, reward + 10.0, reward)
        return reward.astype(np.float32)

    def set_success_dis(self, value):
        self._success_dis = value
        self.env.success_dis = value


def make_env(success_dis=0.1):
    env = SoArm101ReachEnv(render_mode=None)
    env.success_dis = success_dis
    env = ReachDictWrapper(env, success_dis=success_dis)
    return env


class GoalPreservingVecNormalize(VecNormalize):
    """VecNormalize 的修改版：只归一化 observation key，不归一化 goal keys"""
    def normalize_obs(self, obs):
        """只归一化 'observation' key，保持 goal 不变"""
        if isinstance(obs, dict):
            result = {}
            for key, value in obs.items():
                if key == "observation" and self.norm_obs:
                    # 对 observation key 做标准归一化
                    result[key] = (value - self.obs_rms[key].mean) / np.sqrt(self.obs_rms[key].var + 1e-8)
                else:
                    # goal keys 保持原始值
                    result[key] = value
            return result
        return super().normalize_obs(obs)


def train_stage(model, stage_name, success_dis, total_steps, eval_callback):
    """训练一个阶段"""
    print(f"\n{'='*60}")
    print(f"阶段: {stage_name} (success_dis={success_dis}m)")
    print(f"{'='*60}")

    # 更新所有环境的成功阈值
    for env in model.get_env().envs:
        if hasattr(env, 'set_success_dis'):
            env.set_success_dis(success_dis)
        elif hasattr(env, 'success_dis'):
            env.success_dis = success_dis

    model.learn(
        total_timesteps=total_steps,
        callback=eval_callback,
        reset_num_timesteps=False,
    )

    # 评估
    eval_callback.best_mean_reward = -float('inf')
    return model


def main():
    print("=" * 60)
    print("SAC + HER + Curriculum Learning")
    print("=" * 60)

    success_dis_stages = [0.1, 0.05, 0.02]
    steps_per_stage = [500000, 1000000, 3500000]  # 0.02m最难，给最多步数

    # 1. 创建环境（初始用 0.1m）
    print("\n1. Creating environments...")
    train_env = DummyVecEnv([lambda: make_env(success_dis=0.1) for _ in range(8)])
    train_env = GoalPreservingVecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([lambda: make_env(success_dis=0.1)])
    eval_env = GoalPreservingVecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

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
        best_model_save_path="./checkpoints/sac_her_curriculum/",
        log_path="./logs/my_env/sac_her_curriculum/",
        eval_freq=20000,
        n_eval_episodes=20,
        deterministic=True,
    )

    # 4. 分阶段训练
    print("4. Starting curriculum training...")
    total_trained = 0
    for i, (sd, steps) in enumerate(zip(success_dis_stages, steps_per_stage)):
        stage_name = f"Stage {i+1}: success_dis={sd}m"
        model = train_stage(model, stage_name, sd, steps, eval_callback)
        total_trained += steps

        # 更新 eval_env 的阈值
        for env in eval_env.envs:
            if hasattr(env, 'set_success_dis'):
                env.set_success_dis(sd)
            elif hasattr(env, 'success_dis'):
                env.success_dis = sd

    # 5. 保存
    print("\n5. Saving model...")
    model.save("models/sac_her_curriculum_reach")
    train_env.save("vec_normalize/vec_normalize_sac_curriculum.pkl")
    print(f"   Saved to: sac_her_curriculum_reach.zip")
    print(f"   Total steps: {total_trained}")

    train_env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
