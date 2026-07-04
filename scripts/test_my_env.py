# -*- coding: utf-8 -*-
"""测试你自己写的环境"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

# 导入你自己写的环境
from envs.Mso_arm101_reach import SoArm101ReachEnv

print("=" * 60)
print("Testing Your Environment")
print("=" * 60)

# 1. 创建环境
print("\n1. Creating environment...")
env = SoArm101ReachEnv(render_mode=None)

# 2. 测试 reset
print("2. Testing reset...")
obs, info = env.reset()
print(f"   Observation shape: {obs.shape}")
print(f"   Joint angles (deg): {np.degrees(obs[:6])}")
print(f"   End effector pos: {obs[6:9]}")
print(f"   Target pos: {obs[9:12]}")

# 计算初始距离
ee_pos = obs[6:9]
target_pos = obs[9:12]
init_distance = np.linalg.norm(ee_pos - target_pos)
print(f"   Initial distance: {init_distance:.4f} m")

# 3. 测试 step
print("\n3. Testing step with random actions...")
total_reward = 0
for i in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if (i + 1) % 10 == 0:
        ee_pos = obs[6:9]
        target_pos = obs[9:12]
        dist = np.linalg.norm(ee_pos - target_pos)
        print(f"   Step {i+1}: reward={reward:.2f}, distance={dist:.4f}m")

print(f"\n   Total reward (50 steps): {total_reward:.2f}")

# 4. 测试多个episode
print("\n4. Testing multiple episodes...")
success_count = 0
for episode in range(10):
    obs, _ = env.reset()
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    ee_pos = obs[6:9]
    target_pos = obs[9:12]
    distance = np.linalg.norm(ee_pos - target_pos)
    success = distance < 0.02
    if success:
        success_count += 1
    print(f"   Episode {episode+1}: distance={distance:.4f}m, {'SUCCESS' if success else 'FAILED'}")

print(f"\n   Success rate: {success_count}/10 ({success_count*10}%)")

env.close()
print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
