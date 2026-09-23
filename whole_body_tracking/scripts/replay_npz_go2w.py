"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_npz_go2w.py --motion_file path/to/motion.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
# parser.add_argument("--registry_name", type=str, required=True, help="The name of the wand registry.")
parser.add_argument("--motion_file", type=str, required=True, help="Path to the motion .npz file.")
parser.add_argument("--video", action="store_true", default=False, help="Record the replay to an MP4 file.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in replay frames.")
parser.add_argument(
    "--video_dir", type=str, default="videos/replay_npz_go2w", help="Directory in which to save replay videos."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import pathlib

import imageio.v2 as imageio

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.go2w import UNITREE_GO2W_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = UNITREE_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Enabled only when ``--video`` is requested in main().
    camera: CameraCfg | None = None


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # registry_name = args_cli.registry_name
    # if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
    #     registry_name += ":latest"
    # import pathlib

    # import wandb

    # api = wandb.Api()
    # artifact = api.artifact(registry_name)
    motion_file = args_cli.motion_file

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    video_writer = None
    video_frame = 0
    if args_cli.video:
        if args_cli.video_length <= 0:
            raise ValueError(f"--video_length must be positive, got {args_cli.video_length}.")
        video_dir = pathlib.Path(args_cli.video_dir).expanduser().resolve()
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"{pathlib.Path(motion_file).stem}.mp4"
        video_fps = int(np.asarray(motion.fps).item())
        video_writer = imageio.get_writer(video_path, fps=video_fps, codec="libx264")
        camera = scene["camera"]
        print(f"[INFO]: Recording {args_cli.video_length} frames at {video_fps} FPS to: {video_path}")

    # Simulation loop
    try:
        while simulation_app.is_running():
            time_steps += 1
            reset_ids = time_steps >= motion.time_step_total
            time_steps[reset_ids] = 0

            root_states = robot.data.default_root_state.clone()
            root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins
            root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
            root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
            root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

            robot.write_root_state_to_sim(root_states)
            robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
            scene.write_data_to_sim()

            camera_targets = root_states[:, :3]
            camera_eyes = camera_targets + torch.tensor([2.0, 2.0, 0.5], device=sim.device)
            if args_cli.video:
                camera.set_world_poses_from_view(camera_eyes, camera_targets)

            sim.render()  # We don't want physics (sim.step()).
            scene.update(sim_dt)

            pos_lookat = camera_targets[0].cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

            if video_writer is not None:
                rgb_frame = camera.data.output["rgba"][0, :, :, :3].cpu().numpy()
                video_writer.append_data(rgb_frame)
                video_frame += 1
                if video_frame >= args_cli.video_length:
                    break
    finally:
        if video_writer is not None:
            video_writer.close()
            print(f"[INFO]: Video saved to: {video_path}")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    if args_cli.video:
        scene_cfg.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/ReplayCamera",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgba"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=4.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
        )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
