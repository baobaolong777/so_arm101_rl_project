# -*- coding: utf-8 -*-
"""
可视化训练好的SO-ARM101策略
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import envs
from envs.so_arm101_reach import SOArm101ReachEnv
from stable_baselines3 import PPO
import numpy as np

def main():
    # 1. 加载训练好的模型
    model_path = "ppo_so_arm101_reach"
    if os.path.exists(model_path + ".zip"):
        print(f"Loading model from {model_path}...")
        model = PPO.load(model_path)
    else:
        print("Model not found! Please train first.")
        return

    # 2. 创建带渲染的环境
    print("Creating environment with rendering...")
    env = SOArm101ReachEnv(render_mode="human")

    # 3. 测试10个episode
    print("Running 10 episodes...")
    for episode in range(10):
        obs, _ = env.reset()
        total_reward = 0

        for step in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        ee_pos = obs[6:9]
        target_pos = obs[9:12]
        distance = np.linalg.norm(ee_pos - target_pos)
        success = "SUCCESS" if distance < 0.02 else "FAILED"

        print(f"  Episode {episode+1}: {success}, distance={distance:.4f}m, reward={total_reward:.2f}")

    print("\nPress Enter to close...")
    input()
    env.close()


if __name__ == "__main__":
    main()
