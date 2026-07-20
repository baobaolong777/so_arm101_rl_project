import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from envs.Mso_arm101_reach import SoArm101ReachEnv

def make_env(render_mode = None):
    return SoArm101ReachEnv(render_mode)

def main():
    train_env = SubprocVecEnv([make_env for _ in range(8)])
    eval_env = SubprocVecEnv([make_env])
    model = PPO(
        policy="MlpPolicy",
        verbose= 1,
        env = train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./logs/my_env/",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128])
        ),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/my_env/",
        log_path="./logs/my_env/eval/",
        eval_freq=100000,
        n_eval_episodes=5,
        deterministic=True,
    )

    model.learn(
        total_timesteps=10000000,
        callback= eval_callback
    )

    model.save("PPO_Reach")
    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()