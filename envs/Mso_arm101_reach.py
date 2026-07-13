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

        # 目标位置
        self.target_pos = None
        self.max_steps = 300
        self.current_step = 0
        self.success_dis = 0.02
    
    def _get_valid_target(self) -> np.ndarray:
        """用正向运动学生成可达目标：随机关节角度 → 计算末端位置"""
        # 保存当前状态
        qpos_backup = self.data.qpos[:6].copy()

        for _ in range(100):
            # 随机前5个关节（不含gripper）
            for i in range(5):
                jnt_range = self.model.jnt_range[i]
                self.data.qpos[i] = self.np_random.uniform(jnt_range[0], jnt_range[1])
            self.data.qpos[5] = 0.0  # gripper 闭合

            # 正向运动学计算末端位置
            mujoco.mj_forward(self.model, self.data)
            ee_pos = self.data.site_xpos[self.ee_site_id].copy()

            # 检查是否在合理范围（地面以上 + 远离底座）
            dist_from_base = np.linalg.norm(ee_pos)
            if ee_pos[2] > 0.05 and dist_from_base > 0.1:
                # 恢复状态
                self.data.qpos[:6] = qpos_backup
                mujoco.mj_forward(self.model, self.data)
                return ee_pos.astype(np.float32)

        # fallback
        self.data.qpos[:6] = qpos_backup
        mujoco.mj_forward(self.model, self.data)
        return np.array([0.0, 0.0, 0.25], dtype=np.float32)

    def reset(self,seed=None, options=None):
        super().reset(seed=seed)

        # 重置Mujoco仿真
        mujoco.mj_resetData(self.model,self.data)

        # 生成有效目标位置（在工作空间内）
        self.target_pos = self._get_valid_target()

        # 从 home 位置出发，随机偏移小角度（保证起点靠近目标）
        home = np.array([0.0, -0.5, 0.8, -0.3, 0.0])
        noise = self.np_random.uniform(-0.3, 0.3, size=5)
        self.data.qpos[:5] = home + noise
        self.data.qpos[5] = 0.0  # gripper
        mujoco.mj_forward(self.model, self.data)

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

        # 每个action执行20个物理步（0.04s），让位置控制器有时间驱动关节到位
        physics_steps_per_action = 20
        for _ in range(physics_steps_per_action):
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1
        # 获取观测
        obs = self._get_obs()
        # 计算奖励
        reward = self._compute_reward()

        terminated = self._check_success()  # 到达目标即终止
        truncated = self.current_step >= self.max_steps

        if self.render_mode == "human":
            self.render()

        info = {"is_success": self._check_success()}
        return obs, reward, terminated, truncated, info


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
        """距离惩罚 + 成功奖励 + 夹爪惩罚"""
        ee_pos = self.data.site_xpos[self.ee_site_id]
        distance = np.linalg.norm(ee_pos - self.target_pos)

        # 核心：负距离平方
        reward = -100 * distance ** 2

        # 成功奖励
        if distance < self.success_dis:
            reward += 100.0

        # 夹爪惩罚（鼓励保持闭合）
        gripper_angle = abs(self.data.qpos[5])
        reward -= 2.0 * gripper_angle

        # 步惩罚
        reward -= 0.01

        return reward
    
    def _check_success(self):
        ee_pos = self.data.site_xpos[self.ee_site_id]
        return np.linalg.norm(ee_pos - self.target_pos) < self.success_dis

    def close(self):
        if hasattr(self, "renderer"):
            self.renderer.close()

        