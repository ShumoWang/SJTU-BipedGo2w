import os
import time

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import check_nan

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None


class ResidualMotionOnPolicyRunner(MotionOnPolicyRunner):
    """Train residual actions on top of a frozen flat-tracking policy."""

    object_observation_dim = 7

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        registry_name: str = None,
        base_policy_path: str | None = None,
    ):
        if base_policy_path is None:
            raise ValueError("Residual training requires a TorchScript base policy path.")
        super().__init__(env, train_cfg, log_dir, device, registry_name)
        self.base_policy = torch.jit.load(base_policy_path, map_location=device).eval()
        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)
        self.base_observation_dim = self._base_observation_dim()

    def _base_observation_dim(self) -> int:
        for name, parameter in self.base_policy.named_parameters():
            if name.endswith("mlp.0.weight"):
                return parameter.shape[1]
        raise ValueError("Could not determine the base policy observation dimension from its first MLP layer.")

    def _base_actions(self, obs: torch.Tensor) -> torch.Tensor:
        policy_obs = obs["policy"]
        expected_hoi_dim = self.base_observation_dim + self.object_observation_dim
        if policy_obs.shape[-1] != expected_hoi_dim:
            raise ValueError(
                "Residual HOI policy observation size does not match the base policy: "
                f"expected {expected_hoi_dim} (= base {self.base_observation_dim} + "
                f"object {self.object_observation_dim}), got {policy_obs.shape[-1]}."
            )

        proprioception_dim = self.env.num_actions * 3
        object_start = self.base_observation_dim - proprioception_dim
        base_obs = torch.cat(
            (policy_obs[:, :object_start], policy_obs[:, object_start + self.object_observation_dim :]), dim=-1
        )
        with torch.inference_mode():
            return self.base_policy(base_obs)

    def save(self, path: str, infos=None):
        """Save residual checkpoints through the current RSL-RL logger API."""
        OnPolicyRunner.save(self, path, infos)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    residual_actions = self.alg.act(obs)
                    actions = self._base_actions(obs) + residual_actions
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            stop = time.time()
            self.current_learning_iteration = it
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=stop - start,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()
