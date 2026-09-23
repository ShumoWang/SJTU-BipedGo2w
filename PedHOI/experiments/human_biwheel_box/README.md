# 箱体约束的双前轮托举参考

基于上一版 `human_biwheel/reference.npz` 和同一条原始 OMOMO `sub16_largebox_010`。这次修正的是物体和末端的相对几何关系，不把人体左右手腕独立缩放后直接当目标。

- 试验箱体尺寸 24 × 44 × 18 cm；质量假设 0.5 kg，并非源数据物体质量的测量值。
- 两个前轮真实 mesh 的最高支撑表面分别贴合箱底左右两侧。它们托举箱子，不是假设有手指夹住箱体。
- 保留抬升、运输、放下的顺序；根据机器人可达性映射高度。运输时两前腿保持对称支撑。落箱延至底盘停止平移之后。
- 拾取后增加约 0.34 m 的差速后退，让机器人离开取物支撑面后再转向。增加对应后轮转角，重新检查非完整滚动约束。
- 场景使用固定薄支撑面（用于表示外部固定取放架的平台，不是悬浮的动态物体）。箱子在物理测试里是具有质量、惯性和自由关节的刚体，没有 weld、mocap 或根部支撑力。

`box_reference.mp4` 是 **2 倍速运动学参考预览**，箱子位姿按参考设置；不代表闭环抓持成功。
`report.json` 是全帧几何和后轮滚动检查。
`physics_report.json`、`free_box_rollout.npz` 是独立的自由箱体运输段测试：从运输起点初始化，使用原平衡控制结构、重新计算的前腿前馈和近似负载重力。此测试不是完整取放验证，也不是训练好的 HOI 策略。

上一版的空载动力学通过不能继承为本版带箱通过。运输参考中的箱体位置由基座轨迹和物体阶段共同生成；箱体尺寸、轨迹高度、时序有本体适配，并未声称精确复制原始箱体。

运行（工作目录为项目工作区）：

```bash
OPENBLAS_NUM_THREADS=1 python experiments/human_biwheel_box/build_box.py
OPENBLAS_NUM_THREADS=1 python experiments/human_biwheel_box/physics_check.py
python experiments/human_biwheel_box/validate.py
MUJOCO_GL=egl python experiments/human_biwheel_box/render.py
MUJOCO_GL=egl python experiments/human_biwheel_box/render_physics.py
```

原始 `human_biwheel` 目录不被修改。输出 NPZ 包含机器人 qpos/qvel/qacc、箱体 position/quaternion(wxyz)、尺寸、质量假设和双侧支撑目标，不应直接当作已经验收的训练数据。

本次结果：全片几何检查通过，最大非地面穿透仅约 2.1e-10 m（数值误差），后轮参考接触速度残差峰值 0.000813 m/s。自由箱体运输测试在 2.296 s 后箱体相对目标偏离达到 0.181 m，判定失败。不得将双前轮接触存在率当作抓持稳定性；下一步需要箱体位置/姿态反馈和接触约束控制。失败测试视频为 `free_box_test.mp4`。
