# Go2W 地面夹持拿放参考：sub16_largebox_010

这是 2026-09-23 生成、对应运动学验收通过的最终重定位结果。源序列为 OMOMO `sub16_largebox_010`，机器人为双后轮站立的 Go2W，箱体尺寸 0.24 × 0.32 × 0.26 m，假设质量 0.5 kg。

- [reference.npz](reference.npz)：最终机器人与物体参考，供回放、检查和重定位流程使用。
- [episode.npz](episode.npz)：面向后续训练接入的导出，含格式元数据与验收状态；尚非现有 Isaac MotionLoader 的直接输入。
- [schema.json](schema.json)：坐标、单位与四元数约定。
- [acceptance.json](acceptance.json)、[physics_report.json](physics_report.json)：绑定该参考的验收与物理诊断。
- [sha256.json](sha256.json)：文件完整性校验值。

两个 NPZ 均为 3073 帧、120 Hz、25.6 秒。`qpos` 是 xyz + wxyz + 16 关节；关节顺序见 `joint_names`。`qvel` 的根线速度为世界系、根角速度为本体系。物体四元数为 wxyz，兼容字段 `root_rot` 为 xyzw；新接口优先使用 episode 中的显式 wxyz 字段。

```python
import numpy as np
with np.load("episode.npz", allow_pickle=False) as data:
    print(data.files)
    print(data["qpos"].shape)  # (3073, 23)
```

运动学通过；自由箱体闭环在 4.752 秒触发相对位置误差阈值，负载操作未验证，未训练 RL 策略。不要把参考轨迹当成真实动力学执行结果。该失败也不证明 RL 不可学习。

这些文件包含从人体 HOI 派生的参考，不包含原始数据集、SMPL-X 模型或机器人 mesh。使用时仍需遵循源数据及模型的适用条款。环境准备、模型路径限制和完整运行说明见 [主 README](../../../README.md)。
