# Go2W HOI MuJoCo 测试

该目录已经放好本次策略所需的三个文件：

- `policy.pt`：训练后导出的组合 TorchScript 策略（base + residual）
- `motion.csv`：由训练动作 `motion.npz` 转换得到的参考轨迹
- `config.yaml`：302 维观测、16 维动作、关节顺序及控制参数

## 编译

```bash
cd /home/shumowang/Desktop/mimic_ws/rl_sar
./build.sh -mj
```

若系统提示找不到 GLFW，请先安装 Ubuntu 开发包：

```bash
sudo apt update
sudo apt install libglfw3-dev
```

## 运行

```bash
cd /home/shumowang/Desktop/mimic_ws/rl_sar
./cmake_build/bin/rl_sim_mujoco go2w scene_hoi
```

MuJoCo 窗口打开后，先点击窗口让它获得键盘焦点：

1. 按 `0`，等待 Go2W 完成起身。
2. 按 `2`，进入 HOI 策略；程序会自动把机器人和 `CarryCube` 重置到动作首帧。
3. 按 `R` 可从动作首帧重新开始。
4. 按 `P` 返回被动状态，按 `Ctrl+C` 退出程序。

数字键已经由 rl_sar 优先处理，不会再触发 MuJoCo Viewer 的 `Geom 0`、`Geom 2` 等显示开关；终端窗口中的相同按键仍可使用。

HOI 动作播放结束后会保持最后一帧参考，不会自动循环。

## 更换动作或策略

替换策略时，将 whole_body_tracking 的 combined `exported/policy.pt` 复制为此目录下的 `policy.pt`。不要使用 residual-only 文件或训练 checkpoint `model_<iteration>.pt`。替换动作时执行：

```bash
python3 src/rl_sar/scripts/convert_hoi_motion.py \
  /path/to/motion.npz \
  policy/go2w/whole_body_tracking/motion.csv
```

当前适配要求策略输入为 302 维、输出为 16 维，并且动作文件中的关节顺序与训练时一致。302 维由 95 维 base 观测和 207 维 HOI 观测组成；HOI 部分含当前 cube 相对位姿，以及 `[0.1, 0.2, 0.4, 0.6]` 秒的 future anchor、object 和 joint 特征。MuJoCo 测试使用运动文件中的 object 轨迹；真实机器人仍需接入 cube 位姿感知，才能提供当前 object 特征。
