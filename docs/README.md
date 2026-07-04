# SO-ARM101 强化学习项目

> 使用强化学习训练SO-ARM101机械臂 + Sim2Sim迁移

## 项目结构

```
so_arm101_rl_project/
├── assets/                  # 模型和资源文件
│   ├── so101_new_calib.xml  # 官方MuJoCo模型（推荐）
│   ├── scene.xml            # 场景文件
│   ├── joints_properties.xml # 关节属性
│   └── *.stl               # 3D网格文件
│
├── envs/                    # 自定义Gymnasium环境
│
├── logs/                    # 训练日志
│
├── checkpoints/             # 模型检查点
│
├── configs/                 # 配置文件
│
├── LEARNING_PLAN.md         # 学习计划
├── requirements.txt         # Python依赖
└── README.md                # 本文件
```

## 快速开始

### 1. 激活环境
```bash
conda activate RLStudy
```

### 2. 测试官方模型
```python
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("assets/so101_new_calib.xml")
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch(model, data)
input("Press Enter to close...")
viewer.close()
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
- [Gymnasium Documentation](https://gymnasium.farama.org)
