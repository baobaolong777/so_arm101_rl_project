# -*- coding: utf-8 -*-
"""
用你自己写的环境训练PPO
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from envs.Mso_arm101_reach import SoArm101ReachEnv
from scripts.train_sac_her_curriculum import GoalPreservingVecNormalize


def make_env():
    return SoArm101ReachEnv(render_mode=None)


class CurriculumCallback(BaseCallback):
    """课程学习：根据成功率逐步收紧成功阈值"""
    def __init__(self, eval_env, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.thresholds = [0.1, 0.05, 0.03, 0.02]  # 从宽松到严格
        self.current_idx = 0
        self.success_counts = []
        self.eval_interval = 50000  # 每5万步评估一次

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_interval == 0 and self.current_idx < len(self.thresholds) - 1:
            # 评估当前成功率（向量化环境返回格式：obs, rewards, dones, infos）
            success_count = 0
            n_eval = 20
            for _ in range(n_eval):
                obs = self.eval_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, dones, infos = self.eval_env.step(action)
                    done = dones[0]
                    if infos[0].get("is_success", False):
                        success_count += 1
                        break
            success_rate = success_count / n_eval
            self.success_counts.append(success_rate)

            if self.verbose > 0:
                print(f"\n[Curriculum] threshold={self.thresholds[self.current_idx]:.3f} "
                      f"success_rate={success_rate:.1%}")

            # 成功率>80%时收紧阈值
            if success_rate > 0.8:
                self.current_idx += 1
                new_threshold = self.thresholds[self.current_idx]
                # 更新所有环境的成功阈值
                for env in self.eval_env.envs:
                    env.success_dis = new_threshold
                # 通过training_env更新训练环境
                if hasattr(self.training_env, 'envs'):
                    for env in self.training_env.envs:
                        env.success_dis = new_threshold
                if self.verbose > 0:
                    print(f"[Curriculum] → 收紧到 {new_threshold:.3f}m")
        return True


def main():
    print("=" * 60)
    print("Training PPO with YOUR Environment")
    print("=" * 60)

    # 1. 创建并行环境 + 归一化
    print("\n1. Creating environments...")
    train_env = DummyVecEnv([make_env for _ in range(8)])
    train_env = GoalPreservingVecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env])
    eval_env = GoalPreservingVecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 2. 创建PPO模型
    print("2. Creating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./logs/ppo/",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128])
        ),
    )

    # 3. 评估回调
    print("3. Setting up evaluation...")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/ppo/",
        log_path="./logs/ppo/eval/",
        eval_freq=50000,
        n_eval_episodes=10,
        deterministic=True,
    )

    # 4. 训练
    print("4. Starting training...")
    print("   Total timesteps: 5,000,000")
    print("   Monitor: tensorboard --logdir=./logs/my_env/")
    print()

    model.learn(
        total_timesteps=5000000,
        callback=[eval_callback],
    )

    # 5. 保存模型 + 归一化器
    print("\n5. Saving model...")
    model.save("models/ppo_my_env_reach")
    train_env.save("vec_normalize/vec_normalize_ppo.pkl")
    print("   Saved to: models/ppo_my_env_reach.zip + vec_normalize/vec_normalize_ppo.pkl")

    train_env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
