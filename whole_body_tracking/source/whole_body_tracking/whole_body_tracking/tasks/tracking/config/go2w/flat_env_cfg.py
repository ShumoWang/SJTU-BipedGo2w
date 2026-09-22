from isaaclab.utils import configclass

from whole_body_tracking.robots.go2w import UNITREE_GO2W_CFG, GO2W_ACTION_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg_go2w import TrackingEnvCfg


@configclass
class Go2wFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = GO2W_ACTION_SCALE
        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.body_names = [
            "base_link",
            "FL_hip",
            "FL_calf",
            "FL_thigh",
            "FL_wheel_link",
            "FR_hip",
            "FR_calf",
            "FR_thigh",
            "FR_wheel_link",
            "RL_hip",
            "RL_calf",
            "RL_thigh",
            "RL_wheel_link",
            "RR_hip",
            "RR_calf",
            "RR_thigh",
            "RR_wheel_link",
        ]