> 后续轮足检查：此参考未通过无侧滑运动学筛查，不能作为轮式搬运可行轨迹。检查脚本和定量结果位于 `../wheel_feasibility/`。

# ResMimic carry：Go2W 分阶段参考原型

输入是 ResMimic 的 `legged_gym/assets/motions/carry.pkl`（G1 重定向参考，275 帧，30 Hz），配套 `carry.npz` 只有物体 trans/rot。不是原始人体 SMPL-X。未修改两个来源项目。

- `comparison.gif`：源 G1 骨架、逐帧位置 IK、分阶段修正三栏对比。
- `mesh_comparison.gif`：Go2W 模型逐帧 IK / 修正参考对比。
- `go2w_carry_adapted.pkl`：GMR 风格参考，root_rot 使用 xyzw，dof_pos 是模型的 MuJoCo 关节顺序。
- `go2w_carry_adapted.npz`：用于检查的扩展数据，qpos 的根四元数是 wxyz，另存 root_rot 为 xyzw。不是直接用于 Isaac Lab 的训练 NPZ。
- `go2w_framewise.*`：本实验位置 IK 对照，**不是项目原有 GMR 的输出**。
- `summary.json`：方法、误差和限制。

本实验使用有界位置 IK。人体替身 G1 的手、踝轨迹经尺寸适配作为轮足末端目标；手部前后方向做了显式反射，以适应此片段和直立狗的可达侧。因此这里只检验分阶段参考的思路，不应作为原始人类动作保真度的评估。

取物段保留时变目标；零基帧 110–135（3.67–4.50 秒）采用五次平滑权重切换；之后固定前腿抱持与后腿支撑几何，保留根节点水平路径与朝向。参考轮半径约 0.086 m 来自本地 mesh 尺寸；后轮角度由轮心位移的前向投影积分。源动作末尾保持抱物，没有构造额外放下段。

阶段边界为该片段的暂定人工选择。取物段后足目标最大误差约 7.7 cm。没有施加完整无侧滑约束，没有验证自碰撞、抱物接触、力矩、负载或动态平衡；不会据此声称策略能够搬运。现有训练环境的轮子动作接口也尚未修改。

复现（当前机器 base Python 已具备 NumPy / SciPy / MuJoCo / Matplotlib / ImageIO / Pillow）：

```bash
python build_reference.py
MUJOCO_GL=egl python render_preview.py
```
