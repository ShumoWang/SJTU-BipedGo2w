from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude, matrix_from_quat
from isaaclab.assets import RigidObject

from whole_body_tracking.tasks.tracking.hoi_mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def _box_corners_w(pos_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
    # Return the corners of 0.24 x 0.32 x 0.26 box
    corners_local = torch.tensor(
        [
            [-0.12, -0.16, -0.13],
            [-0.12, -0.16,  0.13],
            [-0.12,  0.16, -0.13],
            [-0.12,  0.16,  0.13],
            [ 0.12, -0.16, -0.13],
            [ 0.12, -0.16,  0.13],
            [ 0.12,  0.16, -0.13],
            [ 0.12,  0.16,  0.13],
        ],
        device=pos_w.device,
        dtype=pos_w.dtype,
    )
    rotation_w = matrix_from_quat(quat_w)
    corners_w = torch.matmul(
        corners_local.unsqueeze(0),
        rotation_w.transpose(-1, -2),
    ) + pos_w.unsqueeze(1)
    return corners_w


def object_point_cloud_distance(
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_name: str = "CarryCube"
) -> torch.Tensor:
    # Find the mean corner distances between simulated and reference box positions
    box: RigidObject = env.scene[asset_name]
    command: MotionCommand = env.command_manager.get_term(command_name)
    box_corners_w = _box_corners_w(
        box.data.root_pos_w, box.data.root_quat_w
    )
    reference_corners_w = _box_corners_w(
        command.object_pos_w, command.object_quat_w
    )
    return torch.linalg.vector_norm(
    box_corners_w - reference_corners_w,
    dim=-1,
    ).mean(dim=-1)


def point_cloud_distance_exp(
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_name: str = "CarryCube"
) -> torch.Tensor:
    return torch.exp(-object_point_cloud_distance(env, command_name, asset_name) * 10)