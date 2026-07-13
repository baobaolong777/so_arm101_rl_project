# -*- coding: utf-8 -*-
"""评估 SAC+HER 模型在 0.02m 阈值下的成功率"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.Mso_arm101_reach import SoArm101ReachEnv
from scripts.train_sac_her_curriculum import GoalPreservingVecNormalize, ReachDictWrapper


def make_env():
    return SoArm101ReachEnv(render_mode=None)


def main():
    n_eval = 100
    success_count = 0
    distances = []

    # 直接用 vec_env 评估（和训练时一致）
    vec_env = DummyVecEnv([lambda: ReachDictWrapper(SoArm101ReachEnv(render_mode=None))])
    vec_env = GoalPreservingVecNormalize.load(
        os.path.join(os.path.dirname(__file__), "..", "vec_normalize", "vec_normalize_sac_curriculum.pkl"),
        vec_env
    )
    vec_env.training = False

    # 加载模型
    model = SAC.load(
        os.path.join(os.path.dirname(__file__), "..", "models", "sac_her_curriculum_reach"),
        env=vec_env
    )

    print(f"Evaluating SAC+HER model at 0.02m threshold ({n_eval} episodes)...")

    for ep in range(n_eval):
        obs = vec_env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            done = dones[0]

        # 从 info 中获取 is_success
        is_success = infos[0].get("is_success", False)
        if is_success:
            success_count += 1

        # 从 obs 中提取距离
        achieved = obs["achieved_goal"][0]
        desired = obs["desired_goal"][0]
        dist = np.linalg.norm(achieved - desired)
        distances.append(dist)

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

    vec_env.close()


if __name__ == "__main__":
    main()
