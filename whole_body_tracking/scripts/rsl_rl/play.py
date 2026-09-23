"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--registry_name", type=str, default=None, help="W&B Registry motion artifact path.")
parser.add_argument(
    "--carry_cube",
    action="store_true",
    default=False,
    help="Spawn a small cube and place it midway between the two front wheels.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video"

if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import os
import pathlib
import re

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

installed_version = metadata.version("rsl-rl-lib")


def _find_motion_npz(download_dir: str) -> pathlib.Path:
    """Resolve a motion NPZ from an artifact without assuming its uploaded filename."""
    root = pathlib.Path(download_dir)
    preferred_path = root / "motion.npz"
    if preferred_path.is_file():
        return preferred_path

    candidates = list(root.rglob("*.npz"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No NPZ motion file was found in artifact directory: {root}")
    raise RuntimeError(f"Multiple NPZ files were found in artifact directory {root}: {candidates}")


def _place_carry_cube_between_front_wheels(env, env_ids=None):
    """Place the play-only cube at the midpoint of the current front-wheel positions."""
    robot = env.scene["robot"]
    carry_cube = env.scene["carry_cube"]
    front_wheel_ids, _ = robot.find_bodies(["FL_wheel_link", "FR_wheel_link"], preserve_order=True)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=robot.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(env_ids, device=robot.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=robot.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    midpoint_w = robot.data.body_pos_w[env_ids][:, front_wheel_ids].mean(dim=1)
    root_pose_w = torch.zeros((env_ids.numel(), 7), device=robot.device)
    root_pose_w[:, :3] = midpoint_w
    root_pose_w[:, 3] = 1.0  # identity quaternion in wxyz convention
    carry_cube.write_root_pose_to_sim(root_pose_w, env_ids=env_ids)
    carry_cube.write_root_velocity_to_sim(torch.zeros((env_ids.numel(), 6), device=robot.device), env_ids=env_ids)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # Convert legacy policy/normalization fields to the installed RSL-RL format.
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # Keep the simulation and policy on the requested devices and use the training seed by default.
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    wandb_run = None
    run_path = None
    if args_cli.wandb_path:
        import wandb

        wandb_path_parts = args_cli.wandb_path.rstrip("/").split("/")
        requested_checkpoint = None
        if re.fullmatch(r"model_\d+\.pt", wandb_path_parts[-1]):
            requested_checkpoint = wandb_path_parts[-1]
            run_path = "/".join(wandb_path_parts[:-1])
        else:
            run_path = "/".join(wandb_path_parts)

        api = wandb.Api()
        wandb_run = api.run(run_path)

        checkpoint_files = [
            file.name for file in wandb_run.files() if re.fullmatch(r"model_\d+\.pt", os.path.basename(file.name))
        ]
        if requested_checkpoint is not None:
            checkpoint_name = requested_checkpoint
        else:
            if not checkpoint_files:
                raise RuntimeError(f"No model_<iteration>.pt checkpoint was found in W&B run '{run_path}'.")
            checkpoint_name = max(
                checkpoint_files, key=lambda name: int(os.path.basename(name).split("_")[1].split(".")[0])
            )

        checkpoint_file = wandb_run.file(checkpoint_name)
        if checkpoint_file is None:
            raise FileNotFoundError(f"Checkpoint '{checkpoint_name}' was not found in W&B run '{run_path}'.")
        download_dir = os.path.join(log_root_path, "temp", wandb_run.id)
        downloaded_file = checkpoint_file.download(download_dir, replace=True)
        resume_path = downloaded_file.name
        downloaded_file.close()

        print(f"[INFO]: Downloaded model checkpoint from: {run_path}/{checkpoint_name}")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # Resolve the reference motion. Explicit CLI inputs take priority over W&B run lineage.
    if args_cli.motion_file is not None:
        motion_path = pathlib.Path(args_cli.motion_file).expanduser().resolve()
        if not motion_path.is_file():
            raise FileNotFoundError(f"Motion file does not exist: {motion_path}")
        env_cfg.commands.motion.motion_file = str(motion_path)
        print(f"[INFO]: Using motion file from CLI: {motion_path}")
    elif args_cli.registry_name is not None:
        import wandb

        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        motion_artifact = wandb.Api().artifact(registry_name)
        motion_path = _find_motion_npz(motion_artifact.download())
        env_cfg.commands.motion.motion_file = str(motion_path)
        print(f"[INFO]: Using motion artifact from Registry: {registry_name}")
    elif wandb_run is not None:
        motion_artifact = next((artifact for artifact in wandb_run.used_artifacts() if artifact.type == "motions"), None)
        if motion_artifact is None:
            raise RuntimeError(
                "The W&B run has no linked motion artifact. Pass --registry_name <artifact> or --motion_file <file>."
            )
        motion_path = _find_motion_npz(motion_artifact.download())
        env_cfg.commands.motion.motion_file = str(motion_path)
    elif not isinstance(env_cfg.commands.motion.motion_file, str):
        raise RuntimeError("A reference motion is required. Pass --registry_name <artifact> or --motion_file <file>.")

    if args_cli.carry_cube:
        env_cfg.scene.carry_cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/CarryCube",
            spawn=sim_utils.CuboidCfg(
                size=(0.27, 0.27, 0.27),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=2.0,
                    dynamic_friction=2.0,
                    restitution=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
            ),
            # It is moved between the front wheels immediately after environment creation.
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.5)),
        )

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    base_env = env.unwrapped

    if args_cli.carry_cube:
        _place_carry_cube_between_front_wheels(base_env)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # load previously trained model
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Export the deterministic policy and its reference motion for deployment.
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
    export_motion_policy_as_onnx(
        env.unwrapped,
        runner.alg.get_policy(),
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, run_path if run_path is not None else "none", export_model_dir)

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)
            if args_cli.carry_cube:
                reset_env_ids = torch.nonzero(dones, as_tuple=False).flatten()
                _place_carry_cube_between_front_wheels(base_env, reset_env_ids)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
