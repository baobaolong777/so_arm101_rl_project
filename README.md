# SO-ARM101 Reach Task - Reinforcement Learning

> 使用 SAC + HER + 课程学习训练 SO-ARM101 机械臂完成精准到达任务

## 结果

| 指标 | 值 |
|------|-----|
| 成功率 (0.02m) | **97%** |
| 训练步数 | 5M (0.5M + 1M + 3.5M) |
| 算法 | SAC + HER + Curriculum Learning |
| 目标生成 | 正向运动学 (FK)，100% 可达 |

## 安装

```bash
# 克隆仓库
git clone https://github.com/baobaolong777/so_arm101_rl_project.git
cd so_arm101_rl_project

# 创建环境
conda create -n rl python=3.10
conda activate rl

# 安装依赖
pip install -r requirements.txt
```

## 使用

### 训练

```bash
# SAC + HER 课程学习（推荐，97% 成功率）
python scripts/train_sac_her_curriculum.py

# PPO 训练（简单但效果一般）
python scripts/train_ppo.py
```

### 评估

```bash
# SAC+HER 评估
python scripts/eval_sac_her.py

# PPO 评估
python scripts/eval_ppo.py
```

### 可视化

```bash
python scripts/visualize.py
```

### TensorBoard

```bash
tensorboard --logdir=./logs
```

## 项目结构

```
so_arm101_rl_project/
├── assets/                         # MuJoCo 模型文件
│   ├── scene.xml                   # 场景文件
│   ├── so101_new_calib.xml         # SO-ARM101 模型
│   └── *.stl                       # 3D 网格文件
│
├── envs/                           # Gymnasium 环境
│   └── Mso_arm101_reach.py         # 到达任务环境
│
├── scripts/                        # 训练和评估脚本
│   ├── train_sac_her_curriculum.py # SAC+HER 课程学习训练
│   ├── train_ppo.py                # PPO 训练
│   ├── eval_sac_her.py             # SAC+HER 评估
│   ├── eval_ppo.py                 # PPO 评估
│   └── visualize.py                # MuJoCo 可视化
│
├── models/                         # 保存的模型 (.zip)
├── vec_normalize/                  # VecNormalize 统计量 (.pkl)
├── checkpoints/                    # 训练 checkpoint
├── logs/                           # TensorBoard 日志
├── docs/                           # 文档
├── requirements.txt
└── README.md
```

## 算法

### SAC + HER + Curriculum Learning

**SAC (Soft Actor-Critic)**
- Off-policy 算法，样本效率高
- 内置熵最大化，探索能力强

**HER (Hindsight Experience Replay)**
- 将失败的 episode 重新标记为成功
- 大幅提高稀疏奖励任务的学习效率

**Curriculum Learning**
- 从易到难分阶段训练
- Stage 1: success_dis = 0.10m
- Stage 2: success_dis = 0.05m
- Stage 3: success_dis = 0.02m

### 目标生成

使用正向运动学 (FK) 生成目标：
```python
# 随机关节角度 → 正向运动学 → 末端位置作为目标
for i in range(5):
    self.data.qpos[i] = np.random.uniform(jnt_range[0], jnt_range[1])
mujoco.mj_forward(self.model, self.data)
ee_pos = self.data.site_xpos[self.ee_site_id]
```

保证目标 100% 可达。

### 奖励函数

```python
reward = -100 * distance ** 2   # 距离惩罚
if distance < success_dis:
    reward += 100.0              # 成功奖励
reward -= 2.0 * abs(gripper)    # 夹爪惩罚（保持闭合）
reward -= 0.01                   # 步惩罚
```

## SO-ARM101 关节配置

| 关节 | 功能 | 角度范围 |
|------|------|----------|
| shoulder_pan | 肩部旋转 | ±110° |
| shoulder_lift | 肩部抬升 | ±100° |
| elbow_flex | 肘部弯曲 | ±97° |
| wrist_flex | 腕部弯曲 | ±95° |
| wrist_roll | 腕部旋转 | ±157° |
| gripper | 夹爪开合 | -10° ~ 100° |

## 参考资源

- [SO-ARM100 GitHub](https://github.com/TheRobotStudio/SO-ARM100)
- [MuJoCo Documentation](https://mujoco.org)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io)
- [Gymnasium](https://gymnasium.farama.org)
- [Hindsight Experience Replay (HER)](https://arxiv.org/abs/1707.01495)

## License

MIT
