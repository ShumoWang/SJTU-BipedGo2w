from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude, matrix_from_quat
from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from whole_body_tracking.tasks.tracking.hoi_mdp.commands import MotionCommand


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


def _box_surface_points_local(
    env: ManagerBasedRLEnv,
    asset_name: str,
    num_points: int = 1024,
    seed: int = 2024,
) -> torch.Tensor:
    """Return a cached, area-uniform point sample on a cuboid's surface."""
    box_cfg = getattr(env.cfg.scene, asset_name, None)
    size = tuple(getattr(getattr(box_cfg, "spawn", None), "size", (0.24, 0.32, 0.26)))
    device = env.scene[asset_name].device
    cache_key = (asset_name, size, num_points, seed, str(device))

    cache = getattr(env, "_hoi_surface_point_cache", None)
    if cache is None:
        cache = {}
        env._hoi_surface_point_cache = cache
    if cache_key in cache:
        return cache[cache_key]

    # Pick faces proportional to their surface area, then sample uniformly on each face.
    size_cpu = torch.tensor(size, dtype=torch.float32)
    half_size = size_cpu * 0.5
    x, y, z = size_cpu.tolist()
    face_areas = torch.tensor([y * z, y * z, x * z, x * z, x * y, x * y])
    face_probabilities = face_areas / face_areas.sum()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    face_ids = torch.multinomial(face_probabilities, num_points, replacement=True, generator=generator)
    points = (torch.rand((num_points, 3), generator=generator) * 2.0 - 1.0) * half_size
    point_ids = torch.arange(num_points)
    axes = torch.div(face_ids, 2, rounding_mode="floor")
    signs = torch.where(face_ids % 2 == 0, -1.0, 1.0)
    points[point_ids, axes] = signs * half_size[axes]

    points = points.to(device=device, dtype=env.scene[asset_name].data.root_pos_w.dtype)
    cache[cache_key] = points
    return points


def _box_surface_points_w(
    points_local: torch.Tensor, pos_w: torch.Tensor, quat_w: torch.Tensor
) -> torch.Tensor:
    """Transform shared cuboid-local points into world coordinates for each environment."""
    rotation_w = matrix_from_quat(quat_w)
    points_w = torch.matmul(points_local.unsqueeze(0), rotation_w.transpose(-1, -2))
    return points_w + pos_w.unsqueeze(1)


def object_point_cloud_distance(
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_name: str = "CarryCube",
        num_points: int = 1024,
        seed: int = 2024,
) -> torch.Tensor:
    """Mean corresponding-surface-point distance between simulated and reference cuboids."""
    box: RigidObject = env.scene[asset_name]
    command: MotionCommand = env.command_manager.get_term(command_name)
    points_local = _box_surface_points_local(env, asset_name, num_points, seed)
    box_points_w = _box_surface_points_w(
        points_local, box.data.root_pos_w, box.data.root_quat_w
    )
    reference_points_w = _box_surface_points_w(
        points_local, command.object_pos_w, command.object_quat_w
    )
    return torch.linalg.vector_norm(box_points_w - reference_points_w, dim=-1).mean(dim=-1)


def point_cloud_distance_exp(
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_name: str = "CarryCube"
) -> torch.Tensor:
    return torch.exp(-object_point_cloud_distance(env, command_name, asset_name) * 10)