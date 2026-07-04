# 注册自定义环境
from gymnasium.envs.registration import register

# 环境
register(
    id="SOArm101-Reach-v0",
    entry_point="envs.so_arm101_reach:SOArm101ReachEnv",
    max_episode_steps=200,
)

# 新写的环境
register(
    id="SOArm101-Reach-MyEnv-v0",
    entry_point="envs.Mso_arm101_reach:SoArm101ReachEnv",
    max_episode_steps=200,
)
