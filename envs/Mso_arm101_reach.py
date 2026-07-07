import numpy as np
import  os
import mujoco
import gymnasium as gym
from gymnasium import spaces

class SoArm101ReachEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    def __init__(self,render_mode = None):
        super().__init__()
        self.render_mode = render_mode

        # 找到模型文件地址
        script_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(script_dir,"..","assets")
        xml_path = os.path.join(assets_dir,"scene.xml")

        # 加载mujoco模型(都有哪几步)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        #判断render_mode
        if self.render_mode == "human":
            self.renderer = mujoco.Renderer(self.model)

        # 末端执行器的site
        self.ee_site_name = "gripperframe"
        self.ee_site_id = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,self.ee_site_name)

        # 目标红球的body ID
        self.target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        # 获取mocap在data中的索引（直接用body_mocapid数组）
        self.target_mocap_pos_id = self.model.body_mocapid[self.target_body_id]
        # 如果不是mocap body，设为-1
        if self.target_mocap_pos_id == -1:
            self.target_mocap_pos_id = None

        # 定义动作空间
        self.action_space = spaces.Box(
            low=-1.0, high=1, shape=(6,), dtype=np.float32
        )
        # 定义观察空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )

        # 工作空间约束
        self.workspace = {
            'x': [-0.3, 0.3],    # x 范围
            'y': [-0.3, 0.3],    # y 范围
            'z': [0.1, 0.5]      # z 范围（桌面以上）
        }

        # 目标位置
        self.target_pos = None
        self.max_steps = 300  # 改为300步
        self.current_step = 0
        self.success_dis = 0.02
    
    def _get_valid_target(self) -> np.ndarray:
        """生成有效目标点（在工作空间内且可达）"""
        max_attempts = 100  # 最大尝试次数
        for _ in range(max_attempts):
            target = self.np_random.uniform(
                low=[self.workspace['x'][0], self.workspace['y'][0], self.workspace['z'][0]],
                high=[self.workspace['x'][1], self.workspace['y'][1], self.workspace['z'][1]]
            )
            # 检查是否可达（简单距离验证）
            if self._is_target_reachable(target):
                return target.astype(np.float32)

        # 如果多次尝试都失败，返回一个默认可达点
        return np.array([0.0, 0.0, 0.3], dtype=np.float32)

    def _is_target_reachable(self, target_pos: np.ndarray) -> bool:
        """检查目标点是否可达（简单距离验证）"""
        # 假设机械臂臂长约为 0.4m
        min_reach = 0.05  # 最小距离（太近会碰撞）
        max_reach = 0.45  # 最大距离（臂长限制）

        distance = np.linalg.norm(target_pos)

        # 检查是否在地面以上
        if target_pos[2] < 0.02:  # 至少 2cm 在地面以上
            return False

        return min_reach <= distance <= max_reach

    def reset(self,seed=None, options=None):
        super().reset(seed=seed)

        # 重置Mujoco仿真
        mujoco.mj_resetData(self.model,self.data)

        # 生成有效目标位置（在工作空间内）
        self.target_pos = self._get_valid_target()

        # 随机初始化关节角度，确保末端在地面以上
        max_attempts = 100
        for attempt in range(max_attempts):
            # 随机初始化关节角度（只控制前5个关节，不包括gripper）
            for i in range(5):  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
                joint_range = self.model.jnt_range[i]
                self.data.qpos[i] = self.np_random.uniform(joint_range[0],joint_range[1])

            # 固定 gripper 为张开状态
            self.data.qpos[5] = 0.0  # gripper 关节

            # 更新关节后必须调用 mj_forward 更新末端位置
            mujoco.mj_forward(self.model,self.data)

            # 检查末端执行器是否在地面以上
            ee_pos = self.data.site_xpos[self.ee_site_id]
            if ee_pos[2] > 0.02:  # 至少 2cm 在地面以上
                break

            # 如果尝试多次都失败，使用默认的 home 位置
            if attempt == max_attempts - 1:
                # 使用一个安全的默认位置
                self.data.qpos[0] = 0.0      # shoulder_pan
                self.data.qpos[1] = -0.5     # shoulder_lift（稍微向下）
                self.data.qpos[2] = 0.8      # elbow_flex（弯曲）
                self.data.qpos[3] = -0.3     # wrist_flex
                self.data.qpos[4] = 0.0      # wrist_roll
                self.data.qpos[5] = 0.0      # gripper
                mujoco.mj_forward(self.model,self.data)

        self.current_step = 0

        # 更新目标球位置（使用mocap_pos）
        if self.target_mocap_pos_id is not None:
            self.data.mocap_pos[self.target_mocap_pos_id] = self.target_pos

        if self.render_mode =="human":
            self.render()

        return self._get_obs(), {}

    def step(self,action):
        for i in range(6):
            ctrl_low = self.model.actuator_ctrlrange[i,0]
            ctrl_high = self.model.actuator_ctrlrange[i,1]
            self.data.ctrl[i] = ctrl_low + (action[i] + 1.0) * 0.5 * (ctrl_high - ctrl_low)
        
        mujoco.mj_step(self.model,self.data)
        self.current_step += 1
        # 获取观测
        obs = self._get_obs()
        # 计算奖励
        reward = self._compute_reward()

        terminated = self._check_success()
        truncated = self.current_step >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return obs,reward,terminated,truncated,{}


    def render(self):
        # 更新目标球位置（使用mocap_pos）
        if self.target_pos is not None and self.target_mocap_pos_id is not None:
            self.data.mocap_pos[self.target_mocap_pos_id] = self.target_pos

        if self.render_mode == "human":
            self.renderer.update_scene(self.data)
            self.renderer.render()
        elif self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data)
            return self.renderer.render()
    
    def _get_obs(self):
        # 6个关节角度 + 6个关节速度
        joint_angles = self.data.qpos[:6].copy()
        joint_velocities = self.data.qvel[:6].copy()

        ee_pos = self.data.site_xpos[self.ee_site_id].copy()

        # 检查 target_pos 是否已初始化
        if self.target_pos is None:
            raise ValueError("target_pos is None! Did you call reset() first?")

        return np.concatenate([joint_angles, joint_velocities, ee_pos, self.target_pos]).astype(np.float32)
    
    # 奖励函数设计
    def _compute_reward(self):
        """连续梯度距离奖励 + 成功奖励"""
        ee_pos = self.data.site_xpos[self.ee_site_id]
        distance = np.linalg.norm(ee_pos - self.target_pos)

        # 核心：负距离平方 → 连续梯度，越靠近梯度越大
        # distance 0.5m → -25,  0.3m → -9,  0.1m → -1,  0.02m → -0.04
        reward = -100 * distance ** 2

        # 成功奖励（到达 0.02m 以内）
        if distance < self.success_dis:
            reward += 100.0

        # 步惩罚（鼓励尽快到达）
        reward -= 0.01

        return reward
    
    def _check_success(self):
        ee_pos = self.data.site_xpos[self.ee_site_id]
        return np.linalg.norm(ee_pos - self.target_pos) < self.success_dis

    def close(self):
        if hasattr(self, "renderer"):
            self.renderer.close()

        