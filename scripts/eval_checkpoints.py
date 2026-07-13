# -*- coding: utf-8 -*-
"""从训练过程中逐个评估 checkpoint，找到真正在 0.02m 下工作的模型"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from scripts.train_sac_her_curriculum import make_env

def main():
    n_eval = 50

    # 加载 eval 历史
    data = np.load('logs/my_env/sac_her_curriculum/evaluations.npz')
    timesteps = data['timesteps']
    successes = data['successes']

    # 找到 Stage 3 (0.02m) 的 eval 点
    # Stage 3 从 ~1.5M 步开始
    stage3_mask = timesteps >= 1_500_000
    stage3_ts = timesteps[stage3_mask]
    stage3_sr = successes[stage3_mask].mean(axis=1)

    print(f"Stage 3 eval points: {len(stage3_ts)}")
    print()

    # 评估每个 Stage 3 的 eval 点
    # 注意：我们只能评估最终模型，因为中间 checkpoint 没保存 VecNormalize
    # 但我们可以用不同阈值的 VecNormalize 来测试

    # 先用最终模型 + 0.02m 环境测试
    eval_env = DummyVecEnv([lambda: make_env(success_dis=0.02)])
    eval_env = VecNormalize.load("vec_normalize_sac_curriculum.pkl", eval_env)
    eval_env.training = False

    model = SAC.load("sac_her_curriculum_reach", env=eval_env)

    print(f"Testing final model at 0.02m threshold ({n_eval} episodes, deterministic)...")
    success_count = 0
    distances = []

    for ep in range(n_eval):
        obs = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            done = dones[0]

        achieved = obs["achieved_goal"][0]
        desired = obs["desired_goal"][0]
        dist = np.linalg.norm(achieved - desired)
        distances.append(dist)
        if dist < 0.02:
            success_count += 1

        if (ep + 1) % 10 == 0:
            print(f"  [{ep+1}/{n_eval}] Success: {success_count}/{ep+1} ({success_count/(ep+1):.0%}) | dist: {dist:.4f}m")

    print(f"\nFinal model @ 0.02m: {success_count}/{n_eval} = {success_count/n_eval:.1%}")
    print(f"  Avg dist: {np.mean(distances):.4f}m ± {np.std(distances):.4f}m")
    print(f"  Min: {min(distances):.4f}m | Max: {max(distances):.4f}m")

    # 也测试 0.05m 和 0.1m 看模型是否工作
    for threshold in [0.05, 0.1]:
        eval_env2 = DummyVecEnv([lambda sd=threshold: make_env(success_dis=sd)])
        eval_env2 = VecNormalize.load("vec_normalize_sac_curriculum.pkl", eval_env2)
        eval_env2.training = False

        model2 = SAC.load("sac_her_curriculum_reach", env=eval_env2)

        sc = 0
        for ep in range(20):
            obs = eval_env2.reset()
            done = False
            while not done:
                action, _ = model2.predict(obs, deterministic=True)
                obs, reward, dones, infos = eval_env2.step(action)
                done = dones[0]
            achieved = obs["achieved_goal"][0]
            desired = obs["desired_goal"][0]
            if np.linalg.norm(achieved - desired) < threshold:
                sc += 1
        print(f"\nFinal model @ {threshold}m: {sc}/20 = {sc/20:.0%}")
        eval_env2.close()

    eval_env.close()

if __name__ == "__main__":
    main()
