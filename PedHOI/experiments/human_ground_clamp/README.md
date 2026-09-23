# Human → ground pick / side clamp / bi-wheel carry / ground place

这版落实“左右夹持 + 从地上拿起再放到地上 + 尽量保留人体参考”。取放架、额外后退段和人工固定的搬运高度均不再使用。上一版托底参考和高台夹持参考保留供对照，不作为本版输出。

输入仍为原始 OMOMO `sub16_largebox_010` 的 SMPL-X 人体和物体数据，没有 G1 中间动作。`source_object.npz` 来自该序列 `obj_rot / obj_com_pos / obj_scale` 与官方 captured-object mesh，供同步对照使用。

## 与人体参考的对应

- 不再把 object COM 超过 15 cm 的粗阈值当成接触事件：源数据约第 34 帧进入明显抬升，第 153 帧仍高于静置高度 2.5 cm；重定向在第 31 帧前保持贴地，第 156 帧后贴地、松开。
- 箱子离地高度 = 原始箱子 COM 离静置高度的增量 × 0.43。运输阶段保留原曲线的升降，不再人为替换成水平平台。静置端点的毫米级源噪声归零。
- 躯干前倾来自人体 pelvis → neck 方向，在机器人中缩放为 0.65 倍并限制到 0–52°。这保留了人放箱时骨盆较高、躯干前倾较大的特点。
- 基座高度使用人体骨盆高度变化，结合前后腿 IK 解算。两后轮始终接地，后轮髋关节保持零外展；改变躯干和腿角时同步补偿轮子相位。
- 时间仍为原始 6.4 s → 机器人 25.6 s，所有相位使用同一时间缩放。

## 夹持和地面约束

测试箱体为 24 × 32 × 26 cm，质量假设 0.5 kg。双前轮内侧接触箱体左右侧的靠身体一端的上角，轮中心比箱顶高约 4 cm，夹持点相对箱体中心向后 9 cm。这样可以保留双侧夹持，并让小腿从箱体后上方接近，避免穿入箱子。末端不是新增的夹爪。

`scene.xml` 使用原机器人模型、原地面和一个自由关节箱体，无取放台、weld、mocap 或额外根部支撑。几何参考渲染会逐帧设置机器人和箱体姿态；物理测试单独运行，不使用这种重设。

全帧检查：起止箱底高度精确为 0；关节不超限；机器人、自身、箱子与地面无有意义的穿透；闭合阶段侧面间隙最大约 0.41 mm；后轮参考接触速度残差峰值约 0.000813 m/s。`mj_geomDistance` 在个别近共面 mesh-box 接触中返回与接触流形不一致的负深度，因此存在接触对时用实际 narrow-phase 接触流形的距离；无接触对时用距离查询。此规则在 `validate.py` 中明确实现。

## 仍然保留的适配和限制

这不是逐坐标复制人体：箱体尺寸有适配，水平运动仍使用上一版满足滚动约束的轨迹，前伸距离是人体物体相对骨盆的前向距离缩放后限制在可达区间，箱体 yaw 跟随机器人而未保留人体物体的全部 roll/pitch/yaw。人体手腕位置没有被直接当作机器人末端位置；接触约束映射到上述箱体上角。`alignment.png` 展示保留下来的高度曲线和前倾关系。

**运动学重定向通过不等于已经靠接触力拿起。** 原有平衡控制结构加上夹持前馈的全程自由箱体测试尚未通过：默认试验在约 4.63 s 的起箱阶段失败，未建立稳定的双侧抓持。箱体质量和摩擦均为试验假设，没有宣称真实机器人可执行或 HOI 策略已经训练完成。参数诊断报告只用于分析，不改变此结论。

## 输出和复现

- `human_ground_clamp.mp4`：人体与机器人同步对照，机器人 4 倍速预览。
- `frame_032.png` / `frame_154.png`：地面取箱和落箱对照。
- `reference.npz`：机器人 qpos/qvel/qacc、箱体 pose、夹紧程度、接触目标、时间与轮接触指标。
- `report.json`：几何验收及保真范围。
- `physics_report.json` / `free_box_rollout.npz`：默认完整自由箱体测试（未通过）。

```bash
OPENBLAS_NUM_THREADS=1 python experiments/human_ground_clamp/build.py
python experiments/human_ground_clamp/validate.py
OPENBLAS_NUM_THREADS=1 python experiments/human_ground_clamp/physics_check.py
python experiments/human_ground_clamp/plot_alignment.py
MUJOCO_GL=egl python experiments/human_ground_clamp/render.py
```

所有动态失败保留在报告中，`training_ready` 和 `payload_manipulation_validated` 仍为 false。
