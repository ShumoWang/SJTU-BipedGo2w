# BipedGo2w: Human Motion Retargeting and Whole-body Pedipulation

This repository brings together tools and experiments for retargeting human motion to the Unitree Go2W wheeled quadruped, training motion-tracking policies, and exploring object manipulation with its front wheels. The goal is to connect human–object interaction (HOI) motion data with a quadruped robot that balances on and moves with its rear wheels, achieving loco-manipulation without the need for an extra robot arm. 

## What this project includes

### Go2W motion tracking and HOI policy workflow

- A Go2W configuration for Isaac Lab, including the 12 leg joints and 4 wheel joints, flat motion tracking, and a box-interaction task.
- A residual HOI training setup that runs a frozen flat-motion policy on the base observation and learns an interaction residual from the full policy observation.
- Current cube pose features and four future-reference samples (0.1, 0.2, 0.4, and 0.6 seconds): anchor pose, object pose, and joint position/velocity targets.
- HOI-specific object tracking, surface-point-cloud rewards, contact sensing, task termination criteria, and a configurable virtual object wrench used as a training aid. Playback can disable that wrench for evaluation.
- A combined TorchScript export that packages the frozen base and residual policies for inference; the full HOI policy observation is 302D (95D base + 207D HOI) and the Go2W action is 16D.
- Motion conversion, replay, and video utilities under `whole_body_tracking/scripts/`.

### PedHOI human-to-robot retargeting

`PedHOI/` contains a workflow for converting human and object trajectories into a Go2W reference for a ground-level pick-and-place motion. It combines SMPL-X human reconstruction, rear-wheel rolling constraints, front-wheel object-contact geometry, interaction-graph-based object-pose refinement, and independent geometry/rolling checks. It also runs a free-box MuJoCo dynamics diagnostic and records reports and provenance.

The checked-in PedHOI reports are snapshots for a single OMOMO sequence. They document a kinematically accepted reference, but also a failed full-duration free-box dynamics rollout. See [PedHOI/README.md](PedHOI/README.md) for scope, dependencies, data requirements, and limitations.

### General Motion Retargeting

`GMR/` contains the General Motion Retargeting implementation and Go2W robot assets used by the retargeting experiments. Refer to [GMR/README.md](GMR/README.md) and [GMR/DOC.md](GMR/DOC.md) for its own installation and usage instructions.

## Installation

Install IsaacLab v2.3.2, following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation.html#). After installing, use the virtual environment that has IsaacLab installed to run following command to install required packages for whole_body_tracking: 

```bash
conda activate your_environment
cd /path/to/BipedGo2w
python -m pip install -e whole_body_tracking/source/whole_body_tracking
```

The motion-tracking and HOI commands below also need access to a valid reference-motion NPZ or W&B Registry artifact. PedHOI has additional SMPL-X, OMOMO, mesh, MuJoCo, and Python-environment dependencies; see its README before running that pipeline.

GMR requires a separate virtual envrionment. See its README for further guide. 

## Train a Go2W policy

### Flat motion-tracking policy

From the `whole_body_tracking/` directory, with the Isaac Lab environment activated:

```bash
python scripts/rsl_rl/train.py \
	--task Tracking-Flat-Go2w-v0 \
	--registry_name <entity>/wandb-registry-motions/<motion>:<version> \
	--headless
```

### HOI residual policy

HOI residual training uses a pretrained Go2W flat policy as its frozen base. Supply the reference motion and the base TorchScript policy:

```bash
python scripts/rsl_rl/train.py \
	--task Tracking-HOI-Go2w-v0 \
	--registry_name <entity>/wandb-registry-motions/<motion>:<version> \
	--base_policy_path <path-to-flat-policy.pt> \
	--headless
```

The exact command-line options can vary with the installed training script; use `--help` if your checkout has different CLI arguments. Training outputs and checkpoints are written under `whole_body_tracking/logs/rsl_rl/`.

## Play and export a policy

Play a flat policy by pointing to its run and checkpoint:

```bash
python scripts/rsl_rl/play.py \
	--task Tracking-Flat-Go2w-v0 \
	--load_run <run-directory> \
	--checkpoint <checkpoint.pt> \
	--num_envs 1
```

For an HOI residual policy, pass the matching base policy and motion artifact:

```bash
python scripts/rsl_rl/play.py \
	--task Tracking-HOI-Go2w-v0 \
	--load_run <hoi-run-directory> \
	--checkpoint <checkpoint.pt> \
	--registry_name <entity>/wandb-registry-motions/<motion>:<version> \
	--base_policy_path <path-to-matching-flat-policy.pt> \
	--num_envs 1
```

To record a video, add `--video --video_length 500`. `video_length` is measured in policy/environment steps. With the current 0.02-second policy period, 500 steps are approximately 10 seconds. To evaluate without the training-time virtual object wrench, add `--disable_virtual_object_force`.

When a base policy is supplied, `play.py` exports a combined `exported/policy.pt` containing both the frozen base and HOI residual. Use that combined artifact for single-model deployment. A residual-only policy or an RSL-RL training checkpoint is not a substitute for the combined model.

## Run an HOI policy in rl_sar with MuJoCo (Sim2Sim)

The `rl_sar/` integration can run the exported Go2W HOI policy in MuJoCo without ROS or ROS 2. It uses the MuJoCo robot and `CarryCube` states directly to construct the same 302-dimensional observation used during Isaac Lab training. The checked-in `rl_sar/policy/go2w/whole_body_tracking/` directory already contains an example policy, reference motion, and configuration.

### 1. Prepare the policy and reference motion

An HOI deployment requires these matching files:

```text
rl_sar/policy/go2w/whole_body_tracking/
├── policy.pt    # combined base + residual TorchScript policy
├── motion.csv   # the reference motion used by this policy
└── config.yaml  # observation, action, controller, and joint settings
```

After running the Isaac Lab `play.py` command above, copy the combined policy—not `residual_policy.pt` and not an RSL-RL `model_*.pt` checkpoint:

```bash
cd /path/to/BipedGo2w
cp whole_body_tracking/logs/rsl_rl/<run>/exported/policy.pt \
  rl_sar/policy/go2w/whole_body_tracking/policy.pt
```

Convert the exact NPZ reference motion used for training/playback to the rl_sar CSV format:

```bash
python3 rl_sar/src/rl_sar/scripts/convert_hoi_motion.py \
  /path/to/motion.npz \
  rl_sar/policy/go2w/whole_body_tracking/motion.csv
```

The converter prints the motion FPS. Set the same value in `rl_sar/policy/go2w/whole_body_tracking/config.yaml` under `motion_fps`; the current example uses 50 Hz. The policy and motion must also use the same 16-joint ordering as the training environment. The supplied configuration expects a 302D observation and a 16D action.

### 2. Build rl_sar with MuJoCo

On Ubuntu, install the GLFW development package if CMake cannot find `glfw3Config.cmake`:

```bash
sudo apt update
sudo apt install libglfw3-dev
```

Build the MuJoCo target:

```bash
cd /path/to/BipedGo2w/rl_sar
./build.sh -mj
```

The build script prepares MuJoCo, the inference runtimes, and the Go2W description files. A ROS environment is not required for this MuJoCo executable.

### 3. Start the HOI simulation

Run the Go2W HOI scene from the `rl_sar/` directory:

```bash
./cmake_build/bin/rl_sim_mujoco go2w scene_hoi
```

`scene_hoi.xml` contains both the Go2W model and the free `CarryCube` object. After the viewer opens, click the MuJoCo window once to give it keyboard focus, then:

| Key | Action |
| --- | --- |
| `0` | Start the Go2W get-up sequence |
| `2` | Enter the HOI policy and reset the robot and cube to reference frame 0 |
| `R` | Restart from reference frame 0 while the HOI policy is active |
| `9` | Enter the controlled get-down sequence |
| `P` | Switch to passive mode |
| `Enter` | Pause or resume the MuJoCo simulation |

Wait for the get-up sequence to finish before pressing `2`. When the reference motion reaches its final frame, rl_sar stops policy inference, captures the measured terminal joint positions, commands zero wheel velocity, and switches to the terminal-pose hold state. Press `2` again to replay the HOI episode from the beginning.

### 4. Record a MuJoCo policy trace (optional)

Pass a third command-line argument to record every policy observation and action:

```bash
mkdir -p tmp
./cmake_build/bin/rl_sim_mujoco go2w scene_hoi tmp/mujoco_trace.csv
```

The CSV is created when HOI policy inference begins after pressing `2`. It can be compared with an Isaac Lab trace using:

```bash
python3 src/rl_sar/scripts/compare_policy_traces.py \
  /path/to/isaac_trace.csv \
  tmp/mujoco_trace.csv
```

MuJoCo Sim2Sim obtains the cube pose directly from simulator state. A real-robot deployment must replace this privileged value with an object-pose estimate from a camera, motion-capture system, or another perception pipeline.

## Replay a retargeted Go2W reference

The PedHOI-generated `episode.npz` can be replayed kinematically in Isaac Sim from the repository root:

```bash
python whole_body_tracking/scripts/replay_npz_go2w.py \
	--motion_file PedHOI/data/retargeted/omomo_largebox/episode.npz \
	--headless --video --video_length 0 \
	--render_fps 15 --playback_speed 4 \
	--video_dir PedHOI/runs/isaac_replay
```


## Repository layout

```text
BipedGo2w/
├── GMR/                # General Motion Retargeting and Go2W assets
├── PedHOI/             # Human/object-to-Go2W retargeting and validation
├── whole_body_tracking/ # Isaac Lab environments, training, playback, export
├── rl_sar/             # Sim2Sim, Sim2Real Deployment
└── README.md
```

## Results

The success criteria is defined as:

$$
S_i(\tau) =
\begin{cases}
1, & \text{if episode } i \text{ reaches its time limit without failure and } \bar{d}_i < \tau, \\
0, & \text{otherwise}.
\end{cases}
$$

where $\bar{d}_i$ is the mean distance between the simulated object and reference object motion. Thus the success rate over $M$ evaluated episodes is

$$
\mathrm{SuccessRate}(\tau)
= \frac{1}{M}\sum_{i=1}^{M} S_i(\tau) \times 100\%.
$$

The configured thresholds are $\tau \in \{0.05\mathrm{m},\,0.10\mathrm{m}\}$. With the baseline motion policy trained, the success rate with 0.1 m is 85.7%, and the 0.05 m success rate is 30.0%. 

A demo video of the trained baseline policy is available here:

[Demo Video](https://github.com/ShumoWang/SJTU-BipedGo2w/raw/main/results/rl-video-step-0.mp4)

## Acknowledgment

This project builds on and adapts ideas and open-source components from [General Motion Retargeting (GMR)](https://github.com/YanjieZe/GMR), [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking), and [ResMimic](https://resmimic.github.io/). GMR informs the motion-retargeting workflow, BeyondMimic provides the foundation for Isaac Lab motion tracking, and ResMimic informs the residual-learning approach used for HOI policy experiments. Please also consult each upstream project for its license and asset-specific terms.



