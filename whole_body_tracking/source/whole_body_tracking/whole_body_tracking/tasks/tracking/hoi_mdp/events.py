from __future__ import annotations

from isaaclab.assets.rigid_object.rigid_object import RigidObject
import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from whole_body_tracking.tasks.tracking.hoi_mdp.commands import MotionCommand

def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def virtual_object_force(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_name: str,
    command_name: str,
):
    """Apply a decaying PD wrench that guides the object toward its reference pose."""
    del env_ids  # This is scheduled as a global-time event for all environments.
    box: RigidObject = env.scene[asset_name]
    command: MotionCommand = env.command_manager.get_term(command_name)
    cfg = env.cfg.virtual_force

    if cfg is not None:
        progress = min(max(env.common_step_counter / max(cfg.decay_steps, 1), 0.0), 1.0)
        kp = cfg.kp_start + progress * (cfg.kp_end - cfg.kp_start)
        kd = cfg.kd_start + progress * (cfg.kd_end - cfg.kd_start)

        position_error_w = command.object_pos_w - box.data.root_link_pos_w
        force_w = kp * position_error_w - kd * box.data.root_link_lin_vel_w

        quat_error_w = math_utils.quat_mul(command.object_quat_w, math_utils.quat_inv(box.data.root_link_quat_w))
        quat_error_w = torch.where(quat_error_w[:, :1] < 0.0, -quat_error_w, quat_error_w)
        error_axis = quat_error_w[:, 1:4]
        error_axis_norm = torch.linalg.vector_norm(error_axis, dim=-1, keepdim=True)
        error_angle = 2.0 * torch.atan2(error_axis_norm, quat_error_w[:, :1].clamp(min=0.0))
        rotation_error_w = error_axis * (error_angle / error_axis_norm.clamp(min=1.0e-8))
        torque_w = kp * rotation_error_w - kd * box.data.root_link_ang_vel_w

        force_norm = torch.linalg.vector_norm(force_w, dim=-1, keepdim=True).clamp(min=1.0e-8)
        torque_norm = torch.linalg.vector_norm(torque_w, dim=-1, keepdim=True).clamp(min=1.0e-8)
        force_w = force_w * torch.clamp(cfg.max_force / force_norm, max=1.0)
        torque_w = torque_w * torch.clamp(cfg.max_torque / torque_norm, max=1.0)

        # Hold the latest wrench across all physics substeps until the next control-step update.
        wrench_composer = box.permanent_wrench_composer
        wrench_composer.reset()
        wrench_composer.set_forces_and_torques(
            forces=force_w.unsqueeze(1),
            torques=torque_w.unsqueeze(1),
            is_global=True,
        )


