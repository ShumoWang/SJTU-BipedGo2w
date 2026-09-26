import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Tracking-Flat-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Go2wFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-HOI-Go2w-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Go2wHOIEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WFlatPPORunnerCfg",
    },
)