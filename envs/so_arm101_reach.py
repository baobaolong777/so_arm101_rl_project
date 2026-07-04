"""
SO-ARM101 到达任务环境
目标：控制机械臂末端到达指定位置
"""
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os


class SOArm101ReachEnv(gym.Env):
    """
    SO-ARM101 到达任务环境

    观测空间: [6个关节角度, 末端xyz位置, 目标xyz位置] = 15维
    动作空间: 6个关节的目标角度 = 6维
    奖励: 距离目标越近奖励越高，到达目标给额外奖励
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # 找到模型文件的路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(script_dir, "..", "assets")
        xml_path = os.path.join(assets_dir, "so101_new_calib.xml")

        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # 渲染器
        if self.render_mode == "human":
            self.renderer = mujoco.Renderer(self.model)

        # 末端执行器的site
        self.ee_site_name = "gripperframe"
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)

        # 动作空间: 6个关节的目标角度
        # 使用归一化 [-1, 1]，在step中映射到实际ctrlrange
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # 观察空间: 关节角度(6) + 末端位置(3) + 目标位置(3) = 12维
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        # 目标位置
        self.target_pos = None
        self.max_steps = 200
        self.current_step = 0

        # 工作空间范围（根据之前分析的结果估计）
        self.workspace_min = np.array([-0.3, -0.3, 0.0])
        self.workspace_max = np.array([0.4, 0.3, 0.4])

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)

        # 重置MuJoCo仿真
        mujoco.mj_resetData(self.model, self.data)

        # 随机化初始关节角度（在范围内的小角度）
        for i in range(self.model.njnt):
            jnt_range = self.model.jnt_range[i]
            mid = (jnt_range[0] + jnt_range[1]) / 2
            self.data.qpos[i] = mid + self.np_random.uniform(-0.3, 0.3)

        # 随机化目标位置（在工作空间内）
        self.target_pos = self.np_random.uniform(self.workspace_min, self.workspace_max)

        self.current_step = 0

        # 执行正向运动学，更新末端位置
        mujoco.mj_forward(self.model, self.data)

        # 渲染
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def step(self, action):
        """执行动作"""
        # 将 [-1, 1] 的action映射到实际ctrlrange
        for i in range(6):
            ctrl_low = self.model.actuator_ctrlrange[i, 0]
            ctrl_high = self.model.actuator_ctrlrange[i, 1]
            self.data.ctrl[i] = ctrl_low + (action[i] + 1.0) * 0.5 * (ctrl_high - ctrl_low)

        # 仿真一步
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1

        # 获取观测
        obs = self._get_obs()

        # 计算奖励
        reward = self._compute_reward()

        # 检查是否完成
        terminated = self._check_success()
        truncated = self.current_step >= self.max_steps

        # 渲染
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        """获取观测"""
        # 6个关节角度
        joint_angles = self.data.qpos[:6].copy()

        # 末端执行器位置
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()

        return np.concatenate([joint_angles, ee_pos, self.target_pos])

    def _compute_reward(self):
        """计算奖励"""
        ee_pos = self.data.site_xpos[self.ee_site_id]
        distance = np.linalg.norm(ee_pos - self.target_pos)

        # 距离奖励（负距离，越近越好）
        reward = -distance * 10.0  # 放大系数，让梯度更明显

        # 成功奖励
        if distance < 0.02:  # 2cm内算成功
            reward += 50.0

        # 每步小惩罚，鼓励尽快完成
        reward -= 0.1

        return reward

    def _check_success(self):
        """检查是否成功"""
        ee_pos = self.data.site_xpos[self.ee_site_id]
        return np.linalg.norm(ee_pos - self.target_pos) < 0.02

    def render(self):
        """渲染"""
        if self.render_mode == "human":
            self.renderer.update_scene(self.data)
            self.renderer.render()
        elif self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data)
            return self.renderer.render()

    def close(self):
        """关闭环境"""
        if hasattr(self, "renderer"):
            self.renderer.close()
