from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
import isaaclab.utils.math as math_utils

from whole_body_tracking.tasks.tracking.hoi_mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)

def cube_root_pos_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("CarryCube")) -> torch.Tensor:
    """Cube root position expressed in the robot root frame."""
    robot: RigidObject = env.scene["robot"]
    cube: RigidObject = env.scene[asset_cfg.name]
    position_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        cube.data.root_pos_w,
        cube.data.root_quat_w,
    )
    return position_b


def cube_root_quat_b(
    env: ManagerBasedEnv,
    make_quat_unique: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("CarryCube"),
) -> torch.Tensor:
    """Cube root orientation expressed in the robot root frame."""
    robot: RigidObject = env.scene["robot"]
    cube: RigidObject = env.scene[asset_cfg.name]
    _, quaternion_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        cube.data.root_pos_w,
        cube.data.root_quat_w,
    )
    return math_utils.quat_unique(quaternion_b) if make_quat_unique else quaternion_b

def future_motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str, times: list) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_pos_list = []

    for time in times:
        pos, _ = subtract_frame_transforms(
            command.robot_anchor_pos_w,
            command.robot_anchor_quat_w,
            command.future_anchor_pos_w(time),
            command.future_anchor_quat_w(time),
        )
        future_pos_list.append(pos)

    return torch.cat(future_pos_list, dim=-1).view(env.num_envs, -1)

def future_motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str, times: list) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_ori_list = []

    for time in times:
        _, ori = subtract_frame_transforms(
            command.robot_anchor_pos_w,
            command.robot_anchor_quat_w,
            command.future_anchor_pos_w(time),
            command.future_anchor_quat_w(time),
        )
        mat = matrix_from_quat(ori)
        ori_6d = mat[..., :2].reshape(mat.shape[0], -1)
        future_ori_list.append(ori_6d)

    return torch.cat(future_ori_list, dim=-1)

def future_object_pos_b(env: ManagerBasedEnv, command_name: str, times: list) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot: RigidObject = env.scene["robot"]

    future_pos_list = []

    for time in times:
        pos, _ = subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            command.future_object_pos_w(time),
            command.future_object_quat_w(time),
        )
        future_pos_list.append(pos)

    return torch.cat(future_pos_list, dim=-1).view(env.num_envs, -1)

def future_object_ori_b(env: ManagerBasedEnv, command_name: str, times: list) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot: RigidObject = env.scene["robot"]

    future_ori_list = []

    for time in times:
        _, ori = subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            command.future_object_pos_w(time),
            command.future_object_quat_w(time),
        )
        mat = matrix_from_quat(ori)
        ori_6d = mat[..., :2].reshape(mat.shape[0], -1)
        future_ori_list.append(ori_6d)

    return torch.cat(future_ori_list, dim=-1)

def future_joint_pos_rel(env: ManagerBasedEnv, command_name: str, times: list) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot: RigidObject = env.scene["robot"]

    future_pos_list = []

    for time in times:
        pos = command.future_joint_pos(time)
        pos = pos - robot.data.joint_pos
        future_pos_list.append(pos)

    return torch.cat(future_pos_list, dim=-1).view(env.num_envs, -1)

def future_joint_vel_rel(env: ManagerBasedEnv, command_name: str, times: list) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot: RigidObject = env.scene["robot"]

    future_vel_list = []

    for time in times:
        vel = command.future_joint_vel(time)
        vel = vel - robot.data.joint_vel
        future_vel_list.append(vel)

    return torch.cat(future_vel_list, dim=-1).view(env.num_envs, -1)