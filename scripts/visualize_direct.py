# -*- coding: utf-8 -*-
"""
使用环境类可视化训练好的策略
显示目标位置（红色球体）
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os

# 获取项目根目录
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)
os.chdir(project_root)

import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from envs.Mso_arm101_reach import SoArm101ReachEnv
import numpy as np
import time


def main():
    # 1. 创建环境（带渲染）
    print("Creating environment...")
    env = SoArm101ReachEnv(render_mode="human")

    # 2. 加载训练好的策略
    print("Loading trained policy...")
    policy_model = PPO.load("scripts/ppo_my_env_reach")

    # 3. 启动MuJoCo viewer
    print("Launching viewer...")
    viewer = mujoco.viewer.launch_passive(env.model, env.data)
    time.sleep(1)

    # 4. 运行10个episode
    for episode in range(10):
        print(f"\n--- Episode {episode + 1} ---")

        # 重置环境（会自动随机target_pos）
        obs, _ = env.reset()

        # 获取目标位置（从环境的target_pos）
        target_pos = env.target_pos.copy()
        print(f"Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")

        # 设置红球位置（使用mocap_pos）
        env.data.mocap_pos[0] = target_pos
        mujoco.mj_forward(env.model, env.data)

        viewer.sync()
        time.sleep(0.5)

        # 运行200步
        for step in range(200):
            # 使用环境的step方法
            action, _ = policy_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # 更新红球位置
            env.data.mocap_pos[0] = target_pos

            # 同步渲染
            viewer.sync()
            time.sleep(0.01)

            if terminated or truncated:
                break

        # 最终距离
        ee_pos = obs[6:9]
        distance = np.linalg.norm(ee_pos - target_pos)
        print(f"  Final distance: {distance:.4f}m")

    print("\nDone! Closing viewer...")
    viewer.close()
    env.close()


if __name__ == "__main__":
    main()
