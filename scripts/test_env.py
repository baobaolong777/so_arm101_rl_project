# -*- coding: utf-8 -*-
"""测试SO-ARM101环境"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import envs
from envs.so_arm101_reach import SOArm101ReachEnv
import numpy as np

# 创建环境（不带渲染，用于测试）
print("Creating SOArm101-Reach-v0 environment...")
env = SOArm101ReachEnv(render_mode=None)

# 测试reset
print("Testing reset...")
obs, info = env.reset()
print(f"  Observation shape: {obs.shape}")
print(f"  Joint angles (deg): {np.degrees(obs[:6])}")
print(f"  End effector pos: {obs[6:9]}")
print(f"  Target pos: {obs[9:12]}")

# 计算初始距离
ee_pos = obs[6:9]
target_pos = obs[9:12]
init_distance = np.linalg.norm(ee_pos - target_pos)
print(f"  Initial distance to target: {init_distance:.4f} m")

# 测试step
print("\nTesting step with random actions...")
total_reward = 0
for i in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if (i + 1) % 10 == 0:
        ee_pos = obs[6:9]
        target_pos = obs[9:12]
        dist = np.linalg.norm(ee_pos - target_pos)
        print(f"  Step {i+1}: reward={reward:.2f}, distance={dist:.4f}m")

print(f"\n  Total reward (50 steps): {total_reward:.2f}")

# 测试成功条件
print("\nTesting success condition...")
obs, _ = env.reset()
ee_pos = obs[6:9]
target_pos = obs[9:12]
print(f"  ee_pos: {ee_pos}")
print(f"  target: {target_pos}")
success = env._check_success()
print(f"  Success: {success}")

env.close()
print("\nTest complete!")
