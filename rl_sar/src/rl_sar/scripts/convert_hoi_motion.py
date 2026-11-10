#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


REQUIRED_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "object_pos_w",
    "object_quat_w",
    "object_lin_vel_w",
    "object_ang_vel_w",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an IsaacLab HOI motion NPZ to rl_sar CSV format.")
    parser.add_argument("input", type=Path, help="Input motion .npz")
    parser.add_argument("output", type=Path, help="Output motion .csv")
    parser.add_argument("--anchor-body-index", type=int, default=0, help="Reference anchor body index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion = np.load(args.input)
    missing_keys = [key for key in REQUIRED_KEYS if key not in motion]
    if missing_keys:
        raise KeyError(f"Missing motion arrays: {', '.join(missing_keys)}")

    frame_count = motion["joint_pos"].shape[0]
    arrays = (
        motion["joint_pos"],
        motion["joint_vel"],
        motion["body_pos_w"][:, args.anchor_body_index],
        motion["body_quat_w"][:, args.anchor_body_index],
        motion["body_lin_vel_w"][:, args.anchor_body_index],
        motion["body_ang_vel_w"][:, args.anchor_body_index],
        motion["object_pos_w"],
        motion["object_quat_w"],
        motion["object_lin_vel_w"],
        motion["object_ang_vel_w"],
    )
    if any(array.shape[0] != frame_count for array in arrays):
        raise ValueError("All motion arrays must have the same frame count")

    output = np.concatenate(arrays, axis=1).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, output, delimiter=",", fmt="%.9g")
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    print(f"Wrote {frame_count} frames at {fps:g} Hz with {output.shape[1]} values per frame to {args.output}")


if __name__ == "__main__":
    main()
