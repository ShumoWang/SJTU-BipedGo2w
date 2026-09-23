> 后续轮足检查：此参考未通过无侧滑运动学筛查，不能作为轮式搬运可行轨迹。检查脚本和定量结果位于 `../wheel_feasibility/`。

# ResMimic kneel：拿起、搬运、放下参考实验

选用本机 ResMimic `legged_gym/assets/motions/kneel.pkl` 与 `kneel.npz`。
PKL 为已重定向到 G1 的参考，不是原始人体 SMPL-X；NPZ 为物体位姿。
250 帧、30 Hz、8.3 秒。物体原点高度从 0.012 m 上升到最高 0.573 m，最终回到 0.011 m；水平起终位移约 1.49 m。双手在放下后收回。

## 分阶段修改

- 0–84 帧：拿取与抬起。
- 85–100 帧（2.83–3.33 s）：五次平滑过渡至固定抱持和后足支撑姿态。
- 100–140 帧（3.33–4.67 s）：固定抱持搬运，保留根节点水平轨迹与朝向，用后轮滚动参考替代步态。
- 140–155 帧（4.67–5.17 s）：平滑退出固定姿态。
- 155–214 帧：下降、放置。
- 215–249 帧：释放、收手。

分段为本次样例的人工选择，不是自动接触检测结果。拿放阶段跟踪尺寸适配后的手部和后足目标，狗的机身高度下限为 0.40 m；不像上一条 carry，本条手部在朝向坐标系中朝前，因此没有使用前后反射。

使用位置 IK，关节满足模型范围，相邻帧非轮关节变化不超过 0.12 rad（30 Hz 下约 3.6 rad/s）。轮子角度根据轮心前向位移及约 0.086 m mesh 半径积分。该限制只保证位置增量有界，未保证力矩、加速度或动力学可行。

## 产物

- `mesh_comparison.gif`：逐帧 IK 对照与分阶段修正的 Go2W 模型回放。
- `comparison.gif`：源 G1 骨架（紫色方点为物体原点）、逐帧 IK、分阶段修正三栏回放。
- `go2w_carry_adapted.pkl`：GMR 风格参考；root_rot 是 xyzw，dof_pos 按 MuJoCo 关节顺序。
- `go2w_carry_adapted.npz`：检查格式，qpos 四元数为 wxyz，root_rot 为 xyzw；包含 body_pos_w、joint_names、carry_blend 和末端误差。不是 Isaac Lab 训练 NPZ。
- `go2w_framewise.*`：此实验的逐帧位置 IK 对照，不是项目原有 GMR 重定向结果。
- `summary.json`：匹配误差、连续性检查和方法限制。

## 运行

```bash
python build_reference.py
MUJOCO_GL=egl python render_preview.py
```

仅运动学参考原型。没有声称实现真实抓取、释放或搬运；未验证负载、动态平衡、碰撞、接触力、侧向无滑移或策略训练。源仓库和上一条 carry 实验均保持不变。
