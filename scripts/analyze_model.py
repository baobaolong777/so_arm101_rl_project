# -*- coding: utf-8 -*-
"""
深入分析SO-ARM101官方模型
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("assets/so101_new_calib.xml")
data = mujoco.MjData(model)

print("=" * 60)
print("SO-ARM101 Model Analysis")
print("=" * 60)

# 1. Basic info
print("\n1. Basic Info")
print("-" * 40)
print(f"   Timestep: {model.opt.timestep} s")
print(f"   Gravity: {model.opt.gravity[2]} m/s^2")
print(f"   Total mass: {sum(model.body_mass):.3f} kg")
print(f"   DOF (degrees of freedom): {model.nq}")
print(f"   Joints: {model.njnt}")
print(f"   Bodies: {model.nbody}")
print(f"   Geoms: {model.ngeom}")
print(f"   Actuators: {model.nu}")

# 2. Joint analysis
print("\n2. Joint Analysis")
print("-" * 40)
for i in range(model.njnt):
    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jnt_range = model.jnt_range[i]
    jnt_axis = model.jnt_axis[i]

    # Damping from dof
    damping = model.dof_damping[i] if i < model.nv else 0

    print(f"\n   [{i}] {jname}")
    print(f"       Axis:  [{jnt_axis[0]:.1f}, {jnt_axis[1]:.1f}, {jnt_axis[2]:.1f}]")
    print(f"       Range: [{np.degrees(jnt_range[0]):.1f} deg, {np.degrees(jnt_range[1]):.1f} deg]")
    print(f"       Damping: {damping:.3f}")

# 3. Actuator analysis
print("\n3. Actuator Analysis")
print("-" * 40)
for i in range(model.nu):
    aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    ctrl_range = model.actuator_ctrlrange[i]
    force_range = model.actuator_forcerange[i]
    kp = model.actuator_gainprm[i, 0]

    print(f"\n   [{i}] {aname}")
    print(f"       Type:     Position control")
    print(f"       kp:       {kp:.2f}")
    print(f"       Ctrl range: [{np.degrees(ctrl_range[0]):.1f} deg, {np.degrees(ctrl_range[1]):.1f} deg]")
    print(f"       Force range: [{force_range[0]:.2f}, {force_range[1]:.2f}] N.m")

# 4. Body (link) analysis
print("\n4. Body (Link) Analysis")
print("-" * 40)
for i in range(model.nbody):
    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    mass = model.body_mass[i]
    inertia = model.body_inertia[i]

    print(f"\n   [{i}] {bname}")
    print(f"       Mass:     {mass:.4f} kg")
    print(f"       Inertia:  [{inertia[0]*1000:.2f}, {inertia[1]*1000:.2f}, {inertia[2]*1000:.2f}] g*cm^2")

# 5. Geom type statistics
print("\n5. Geom Statistics")
print("-" * 40)
type_names = {0: "sphere", 1: "capsule", 2: "ellipsoid", 3: "cylinder",
              4: "box", 5: "plane", 6: "mesh"}
geom_count = {}
for i in range(model.ngeom):
    t = type_names.get(model.geom_type[i], "unknown")
    geom_count[t] = geom_count.get(t, 0) + 1
for t, c in sorted(geom_count.items()):
    print(f"   {t}: {c}")

# 6. Forward kinematics test
print("\n6. Forward Kinematics Test")
print("-" * 40)
print("   Moving all joints to middle position...")

for i in range(model.njnt):
    jnt_range = model.jnt_range[i]
    data.qpos[i] = (jnt_range[0] + jnt_range[1]) / 2

mujoco.mj_forward(model, data)

grip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "baseframe")
if grip_id >= 0:
    ee_pos = data.site_xpos[grip_id]
    print(f"   End effector pos (middle pose): [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}] m")
    print(f"   Distance from base: {np.linalg.norm(ee_pos):.4f} m")

# 7. Workspace test
print("\n7. Workspace Test")
print("-" * 40)
original_qpos = data.qpos.copy()

test_configs = [
    ("All joints = 0", np.zeros(model.njnt)),
    ("All joints = mid", np.array([(model.jnt_range[i, 0] + model.jnt_range[i, 1]) / 2
                                    for i in range(model.njnt)])),
    ("All joints = max", model.jnt_range[:, 1].copy()),
    ("All joints = min", model.jnt_range[:, 0].copy()),
]

for name, qpos in test_configs:
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    if grip_id >= 0:
        ee = data.site_xpos[grip_id]
        print(f"   {name:20s} -> [{ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}] m")

data.qpos[:] = original_qpos
mujoco.mj_forward(model, data)

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)
