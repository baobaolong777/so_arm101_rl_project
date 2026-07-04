# SO-ARM101 强化学习项目

> 使用PPO算法训练SO-ARM101机械臂完成到达任务

## 项目结构

```
so_arm101_rl_project/
│
├── assets/                  # MuJoCo模型文件
│   ├── so101_new_calib.xml  # SO-ARM101官方模型
│   ├── scene.xml            # 场景文件
│   └── *.stl               # 3D网格文件
│
├── envs/                    # Gymnasium环境
│   ├── __init__.py          # 环境注册
│   └── so_arm101_reach.py   # 到达任务环境
│
├── scripts/                 # 可执行脚本
│   ├── train_reach.py       # 训练脚本
│   ├── visualize.py         # 可视化脚本
│   ├── visualize_direct.py  # 直接可视化（带目标显示）
│   ├── test_env.py          # 环境测试
│   ├── analyze_model.py     # 模型分析
│   └── eval_reach.py        # 评估脚本
│
├── checkpoints/             # 训练好的模型
│   └── best_model.zip       # 最佳模型
│
├── logs/                    # TensorBoard日志
│
├── docs/                    # 文档
│   ├── LEARNING_PLAN.md     # 学习计划
│   └── README.md            # 详细说明
│
└── requirements.txt         # Python依赖
```

## 快速开始

### 1. 激活环境
```bash
conda activate RLStudy
```

### 2. 测试环境
```bash
cd so_arm101_rl_project
python scripts/test_env.py
```

### 3. 训练模型
```bash
python scripts/train_reach.py
```

### 4. 可视化结果
```bash
python scripts/visualize_direct.py
```

### 5. 查看训练曲线
```bash
tensorboard --logdir=logs/
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
