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

        # 定义动作空间
        self.action_space = spaces.Box(
            low=-1.0, high=1, shape=(6,), dtype=np.float32
        )
        # 定义观察空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        # 目标位置
        self.target_pos = None
        self.max_steps = 300  # 改为300步
        self.current_step = 0
    
    def reset(self,seed=None, options=None):
        super().reset(seed=seed)

        # 重置Mujoco仿真
        mujoco.mj_resetData(self.model,self.data)

        # 动态设置目标位置
        for i in range(self.model.njnt):
            joint_range = self.model.jnt_range[i]
            self.data.qpos[i] = self.np_random.uniform(joint_range[0],joint_range[1])
        mujoco.mj_forward(self.model,self.data)
        self.target_pos = self.data.site_xpos[self.ee_site_id].copy()
        self.target_pos[2] = max(self.target_pos[2], 0.15)  # z >= 0.15m（更合理的范围）
        # 随机初始化关节角度
        for i in range(self.model.njnt):
            joint_range = self.model.jnt_range[i]
            self.data.qpos[i] = self.np_random.uniform(joint_range[0],joint_range[1])
        
        self.current_step = 0

        mujoco.mj_forward(self.model,self.data)

        # 更新红球位置（在mj_forward之后）
        self.data.xpos[self.target_body_id] = self.target_pos

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
        if self.render_mode == "human":
            self.renderer.update_scene(self.data)
            self.renderer.render()
        elif self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data)
            return self.renderer.render()
    
    def _get_obs(self):
        joint_angles = self.data.qpos[:6].copy()

        ee_pos = self.data.site_xpos[self.ee_site_id].copy()

        return np.concatenate([joint_angles,ee_pos,self.target_pos])
    
    # 奖励函数设计
    def _compute_reward(self):
        """距离惩罚 + 步惩罚"""
        ee_pos = self.data.site_xpos[self.ee_site_id]
        distance = np.linalg.norm(ee_pos - self.target_pos)

        # 距离惩罚
        reward = -distance * 10

        if distance <0.02:
            reward +=100

        # 步惩罚
        reward -= 0.01

        return reward
    
    def _check_success(self):
        ee_pos = self.data.site_xpos[self.ee_site_id]
        return np.linalg.norm(ee_pos - self.target_pos) < 0.02

    def close(self):
        if hasattr(self, "renderer"):
            self.renderer.close()

        