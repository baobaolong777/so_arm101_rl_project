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
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.Mso_arm101_reach import SoArm101ReachEnv
from scripts.train_sac_her_curriculum import GoalPreservingVecNormalize, ReachDictWrapper
import numpy as np
import time


def main():
    # 1. 创建带渲染的环境（用于可视化）
    print("Creating environment...")
    render_env = SoArm101ReachEnv(render_mode="human")

    # 2. 创建训练用的 vec_env（用于策略预测）
    print("Loading trained policy...")
    from stable_baselines3 import SAC

    model_path = os.path.join(project_root, "models", "sac_her_curriculum_reach")
    vec_norm_path = os.path.join(project_root, "vec_normalize", "vec_normalize_sac_curriculum.pkl")

    vec_env = DummyVecEnv([lambda: ReachDictWrapper(SoArm101ReachEnv(render_mode=None))])
    vec_env = GoalPreservingVecNormalize.load(vec_norm_path, vec_env)
    vec_env.training = False

    policy_model = SAC.load(model_path, env=vec_env)

    # 3. 启动MuJoCo viewer
    print("Launching viewer...")
    viewer = mujoco.viewer.launch_passive(render_env.model, render_env.data)
    time.sleep(1)

    # 4. 运行10个episode
    for episode in range(30):
        print(f"\n--- Episode {episode + 1} ---")

        # 重置 vec_env（策略用）
        vec_obs = vec_env.reset()

        # 获取 vec_env 内部的原始环境
        inner_env = vec_env.venv.envs[0].env  # ReachDictWrapper -> SoArm101ReachEnv
        target_pos = inner_env.target_pos.copy()
        print(f"Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")

        # 同步渲染环境的状态
        render_env.target_pos = target_pos.copy()
        mujoco.mj_resetData(render_env.model, render_env.data)
        render_env.data.qpos[:6] = inner_env.data.qpos[:6].copy()
        render_env.data.mocap_pos[0] = target_pos
        mujoco.mj_forward(render_env.model, render_env.data)

        viewer.sync()
        time.sleep(0.5)

        # 运行300步
        final_distance = None
        for step in range(300):
            action, _ = policy_model.predict(vec_obs, deterministic=True)

            # step 之前记录距离（step 后 DummyVecEnv 会自动 reset）
            wrapper_env = vec_env.venv.envs[0]
            raw_env = wrapper_env.env
            ee_pos = raw_env.data.site_xpos[raw_env.ee_site_id]
            final_distance = np.linalg.norm(ee_pos - raw_env.target_pos)

            vec_obs, reward, done, info = vec_env.step(action)

            # 同步渲染环境
            render_env.data.qpos[:6] = raw_env.data.qpos[:6].copy()
            render_env.data.mocap_pos[0] = target_pos
            mujoco.mj_forward(render_env.model, render_env.data)

            viewer.sync()
            time.sleep(0.01)

            if done[0]:
                break

        print(f"  Final distance: {final_distance:.4f}m")

    print("\nDone! Closing viewer...")
    viewer.close()
    render_env.close()
    vec_env.close()


if __name__ == "__main__":
    main()
