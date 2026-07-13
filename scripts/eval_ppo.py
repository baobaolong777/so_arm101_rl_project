# -*- coding: utf-8 -*-
"""评估 PPO 模型在 0.02m 阈值下的成功率"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.Mso_arm101_reach import SoArm101ReachEnv
from scripts.train_sac_her_curriculum import GoalPreservingVecNormalize


def make_env():
    return SoArm101ReachEnv(render_mode=None)


def main():
    n_eval = 100
    success_count = 0
    distances = []

    # 用原始环境评估
    raw_env = make_env()

    # 加载归一化器
    temp_env = DummyVecEnv([make_env])
    vn = GoalPreservingVecNormalize.load(
        os.path.join(os.path.dirname(__file__), "..", "vec_normalize", "vec_normalize_ppo.pkl"),
        temp_env
    )
    vn.training = False

    # 加载模型
    model = PPO.load(
        os.path.join(os.path.dirname(__file__), "..", "models", "ppo_my_env_reach")
    )

    print(f"Evaluating PPO model at 0.02m threshold ({n_eval} episodes)...")

    for ep in range(n_eval):
        obs_raw, _ = raw_env.reset()
        done = False
        truncated = False

        while not done and not truncated:
            obs_norm = vn.normalize_obs(obs_raw.reshape(1, -1))
            action, _ = model.predict(obs_norm, deterministic=True)
            obs_raw, reward, done, truncated, info = raw_env.step(action[0])

        ee_pos = raw_env.data.site_xpos[raw_env.ee_site_id]
        target_pos = raw_env.target_pos
        dist = np.linalg.norm(ee_pos - target_pos)

        distances.append(dist)
        is_success = dist < 0.02
        if is_success:
            success_count += 1

        if (ep + 1) % 20 == 0:
            print(f"  [{ep+1}/{n_eval}] Success rate: {success_count/(ep+1):.1%} | Last dist: {dist:.4f}m")

    final_rate = success_count / n_eval
    avg_dist = np.mean(distances)
    std_dist = np.std(distances)

    print(f"\n{'='*50}")
    print(f"Final Results ({n_eval} episodes, deterministic):")
    print(f"  Success rate: {final_rate:.1%}")
    print(f"  Success count: {success_count}/{n_eval}")
    print(f"  Avg distance: {avg_dist:.4f}m +/- {std_dist:.4f}m")
    print(f"  Min distance: {min(distances):.4f}m")
    print(f"  Max distance: {max(distances):.4f}m")
    print(f"  Median distance: {np.median(distances):.4f}m")
    print(f"{'='*50}")

    raw_env.close()
    temp_env.close()


if __name__ == "__main__":
    main()
