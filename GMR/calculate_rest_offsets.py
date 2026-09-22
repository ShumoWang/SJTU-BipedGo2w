"""Calculate human-SMPL-X -> Unitree Go2 rest-pose rotation offsets.

Convention used here:
    R_go2 = R_human @ R_offset
    R_offset = R_human_rest.T @ R_go2_rest

All rotations are local-link-to-world rotation matrices.
"""

from pathlib import Path

import mujoco
import numpy as np
import torch
import smplx
from smplx.lbs import batch_rodrigues


ROOT = Path(__file__).resolve().parent
SMPLX_PATH = ROOT / "assets" / "body_models"
GO2_XML = ROOT / "assets" / "unitree_go2" / "go2.xml"


# First 22 SMPL-X body joints, in the same order as output.full_pose[:, :22].
HUMAN_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]


# SMPL-X parent indices for the 22 joints above.
HUMAN_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
]


# Change this table if you use a different human-to-Go2 correspondence.
BODY_MAPPING = {
    "pelvis": "base_link",
    "left_hip": "RL_hip",
    "left_knee": "RL_calf",
    "left_foot": "RL_foot",
    "right_hip": "RR_hip",
    "right_knee": "RR_calf",
    "right_foot": "RR_foot",
    "left_shoulder": "FL_hip",
    "left_elbow": "FL_calf",
    "left_wrist": "FL_foot",
    "right_shoulder": "FR_hip",
    "right_elbow": "FR_calf",
    "right_wrist": "FR_foot",
}


def get_human_rest_rotations():
    model = smplx.create(
        model_path=str(SMPLX_PATH),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        batch_size=1,
    )

    with torch.no_grad():
      output = model(return_full_pose=True)

      if output.full_pose is not None:
          pose = output.full_pose.reshape(output.full_pose.shape[0], -1, 3)
          pose = pose[:, :22].reshape(-1, 3)
      else:
          pose = torch.cat(
              [output.global_orient, output.body_pose],
              dim=1,
          )
          pose = pose.reshape(pose.shape[0], -1, 3)
          pose = pose[:, :22].reshape(-1, 3)

      local = batch_rodrigues(pose).reshape(1, 22, 3, 3)[0]


    # Convert local rotations to link-to-world rotations.
    world = torch.empty_like(local)
    for i, parent in enumerate(HUMAN_PARENTS):
        world[i] = local[i] if parent < 0 else world[parent] @ local[i]

    return {name: world[i].cpu().numpy() for i, name in enumerate(HUMAN_NAMES)}


def get_go2_rest_rotations():
    mj_model = mujoco.MjModel.from_xml_path(str(GO2_XML))
    mj_data = mujoco.MjData(mj_model)

    # qpos0 contains the XML-defined default pose, including the free base pose.
    mj_data.qpos[:] = mj_model.qpos0
    mujoco.mj_forward(mj_model, mj_data)

    rotations = {}
    for go2_name in BODY_MAPPING.values():
        body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, go2_name
        )
        if body_id < 0:
            raise ValueError(f"Go2 body not found in XML: {go2_name}")
        rotations[go2_name] = mj_data.xmat[body_id].reshape(3, 3).copy()

    return rotations


def main():
    human_rest = get_human_rest_rotations()
    go2_rest = get_go2_rest_rotations()

    np.set_printoptions(precision=6, suppress=True)
    for human_name, go2_name in BODY_MAPPING.items():
        R_human = human_rest[human_name]
        R_go2 = go2_rest[go2_name]
        R_offset = R_human.T @ R_go2

        print(f"\n{human_name:12s} -> {go2_name}")
        print("R_human_rest =\n", R_human)
        print("R_go2_rest   =\n", R_go2)
        print("R_offset     =\n", R_offset)
        print("check error  =", np.linalg.norm(R_human @ R_offset - R_go2))


if __name__ == "__main__":
    main()
