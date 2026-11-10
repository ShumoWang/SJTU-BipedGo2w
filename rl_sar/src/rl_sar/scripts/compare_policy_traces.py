#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path


OBSERVATION_TERMS = [
    ("motion_command", 32),
    ("motion_anchor_pos_b", 3),
    ("motion_anchor_ori_b", 6),
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("joint_pos", 16),
    ("joint_vel", 16),
    ("last_action", 16),
    ("cube_pos_b", 3),
    ("cube_quat_b", 4),
    ("future_motion_anchor_pos_b", 12),
    ("future_motion_anchor_ori_b", 24),
    ("future_object_pos_b", 12),
    ("future_object_ori_b", 24),
    ("future_joint_pos_rel", 64),
    ("future_joint_vel_rel", 64),
]


def read_trace(path: Path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Trace is empty: {path}")
    observation_names = sorted((name for name in rows[0] if name.startswith("obs_")), key=lambda name: int(name[4:]))
    action_names = sorted((name for name in rows[0] if name.startswith("action_")), key=lambda name: int(name[7:]))
    observations = [[float(row[name]) for name in observation_names] for row in rows]
    actions = [[float(row[name]) for name in action_names] for row in rows]
    jit_action_names = sorted(
        (name for name in rows[0] if name.startswith("jit_action_")), key=lambda name: int(name[11:])
    )
    jit_actions = [[float(row[name]) for name in jit_action_names] for row in rows] if jit_action_names else None
    return observations, actions, jit_actions


def error_stats(left, right):
    errors = [abs(a - b) for left_row, right_row in zip(left, right) for a, b in zip(left_row, right_row)]
    return math.sqrt(sum(error * error for error in errors) / len(errors)), max(errors)


def main():
    parser = argparse.ArgumentParser(description="Compare IsaacLab and MuJoCo policy traces.")
    parser.add_argument("isaac_trace", type=Path)
    parser.add_argument("mujoco_trace", type=Path)
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args()

    isaac_obs, isaac_actions, isaac_jit_actions = read_trace(args.isaac_trace)
    mujoco_obs, mujoco_actions, _ = read_trace(args.mujoco_trace)
    frame_count = min(len(isaac_obs), len(mujoco_obs))
    if args.frames is not None:
        frame_count = min(frame_count, args.frames)
    if frame_count == 0:
        raise ValueError("No common frames to compare")
    if len(isaac_obs[0]) != len(mujoco_obs[0]):
        raise ValueError(f"Observation dimensions differ: {len(isaac_obs[0])} vs {len(mujoco_obs[0])}")
    if len(isaac_actions[0]) != len(mujoco_actions[0]):
        raise ValueError(f"Action dimensions differ: {len(isaac_actions[0])} vs {len(mujoco_actions[0])}")

    isaac_obs = isaac_obs[:frame_count]
    mujoco_obs = mujoco_obs[:frame_count]
    isaac_actions = isaac_actions[:frame_count]
    mujoco_actions = mujoco_actions[:frame_count]
    obs_rmse, obs_max = error_stats(isaac_obs, mujoco_obs)
    action_rmse, action_max = error_stats(isaac_actions, mujoco_actions)
    print(f"frames={frame_count} obs_dim={len(isaac_obs[0])} action_dim={len(isaac_actions[0])}")
    print(f"all_observations: rmse={obs_rmse:.8g} max_abs={obs_max:.8g}")
    print(f"all_actions:      rmse={action_rmse:.8g} max_abs={action_max:.8g}")
    if isaac_jit_actions is not None:
        runner_jit_rmse, runner_jit_max = error_stats(isaac_actions, isaac_jit_actions[:frame_count])
        print(f"isaac_runner_vs_exported_jit: rmse={runner_jit_rmse:.8g} max_abs={runner_jit_max:.8g}")
    print("\nObservation terms:")
    offset = 0
    for name, size in OBSERVATION_TERMS:
        end = offset + size
        left = [row[offset:end] for row in isaac_obs]
        right = [row[offset:end] for row in mujoco_obs]
        rmse, maximum = error_stats(left, right)
        print(f"  {name:<30} [{offset:3d}:{end:3d}] rmse={rmse:.8g} max_abs={maximum:.8g}")
        offset = end
    if offset != len(isaac_obs[0]):
        print(f"  WARNING: term layout covers {offset} of {len(isaac_obs[0])} observations")

    print("\nFirst-frame actions:")
    for index, (isaac_value, mujoco_value) in enumerate(zip(isaac_actions[0], mujoco_actions[0])):
        print(f"  action_{index:02d}: isaac={isaac_value: .8f} mujoco={mujoco_value: .8f} diff={mujoco_value-isaac_value: .8f}")


if __name__ == "__main__":
    main()
