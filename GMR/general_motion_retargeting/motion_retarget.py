
import mink
import mujoco as mj
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=5e-1, # change from 1e-1 to 1e-2.
        verbose: bool=True,
        use_velocity_limit: bool=False,
    ) -> None:

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")
        
        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])
        self.tgt_robot = tgt_robot

        self.max_iter = 10

        self.solver = solver
        self.damping = damping

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 
            
        self.setup_retarget_configuration()
        
        self.ground_offset = 0.0

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)

        # Set a valid initial pose to prevent the go2 knee joint from exceeding its limits starting from 0
        if self.tgt_robot == "unitree_go2":
            home_qpos = np.array([
                0.0, 0.0, 0.4,  # root x,y,z
                1, 0.0, 0.0, 0.0,  # root quaternion w,x,y,z
                0.0, 0.8, -1.5,     # FL
                0.0, 0.8, -1.5,     # FR
                0.0, 1.0, -1.5,     # RL
                0.0, 1.0, -1.5,     # RR
            ], dtype=np.float64)

            self.configuration.data.qpos[:] = home_qpos

        if self.tgt_robot == "unitree_go2w":
            home_qpos = np.array([
                0.0, 0.0, 0.4,  # root x,y,z
                1, 0.0, 0.0, 0.0,  # root quaternion w,x,y,z
                0.0, 0.8, -1.5, 0.0,     # FL
                0.0, 0.8, -1.5, 0.0,    # FR
                0.0, 1.0, -1.5, 0.0,    # RL
                0.0, 1.0, -1.5, 0.0    # RR
            ], dtype=np.float64)

            self.configuration.data.qpos[:] = home_qpos
        #     import mujoco

        #     mujoco.mj_forward(
        #         self.model,
        #         self.configuration.data,
        #     )
        #     body_names = [
        #     "base_link",
        #     "FL_hip",
        #     "FL_calf",
        #     "FL_foot",
        #     "FR_hip",
        #     "FR_calf",
        #     "FR_foot",
        #     "RL_hip",
        #     "RL_calf",
        #     "RL_foot",
        #     "RR_hip",
        #     "RR_calf",
        #     "RR_foot",
        #     ]

        #     for body_name in body_names:
        #         body_id = mujoco.mj_name2id(
        #             self.model,
        #             mujoco.mjtObj.mjOBJ_BODY,
        #             body_name,
        #         )

        #         pos = self.configuration.data.xpos[body_id].copy()

        #         # xmat is world-frame 3x3 rotation matrix
        #         R_world = self.configuration.data.xmat[body_id].reshape(3, 3).copy()

        #         quat = np.zeros(4)
        #         mujoco.mju_mat2Quat(quat, R_world.reshape(-1))

        #         print(f"\n{body_name}")
        #         print("position =", pos)
        #         print("rotation =")
        #         print(R_world)
        #         print("quat wxyz =", quat)

        # stop = input()
                
        self.tasks1 = []
        self.tasks2 = []
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []

        # if self.tgt_robot == "unitree_go2":
        #     geom_pairs = [
        #         (["FL"],
        #         ["FR"]),
        #     ]

        #     self.ik_limits.append(
        #         mink.CollisionAvoidanceLimit(
        #             self.model,
        #             geom_pairs,
        #             gain=0.8,
        #             minimum_distance_from_collisions=0.03,
        #             collision_detection_distance=0.08,
        #         )
        #     )

  
    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)

        if self.tgt_robot == "unitree_go2w":
            base_pos, base_quat = human_data["pelvis"]

            base_rotation = R.from_quat(
                base_quat,
                scalar_first=True,
            )

            arm_mappings = [
                (
                    "FL_hip",
                    "left_shoulder",
                    "left_wrist",
                ),
                (
                    "FR_hip",
                    "right_shoulder",
                    "right_wrist",
                ),
            ]

            # define limits
            min_arm_reach = 0.14
            max_arm_reach = 0.41

            for robot_hip, human_shoulder, human_wrist in arm_mappings:
                shoulder_pos = human_data[human_shoulder][0]
                wrist_pos = human_data[human_wrist][0]

                # find vector pointing from shoulder to wrist
                arm_vector = wrist_pos - shoulder_pos
                arm_length = np.linalg.norm(arm_vector)

                if arm_length > 1e-8:
                    clamped_length = np.clip(
                        arm_length,
                        min_arm_reach,
                        max_arm_reach,
                    )
                    arm_vector = (
                        arm_vector
                        * clamped_length
                        / arm_length
                    )

                # Go2W hip position in the local frame of base_link.
                robot_hip_local = (
                    self.model.body(robot_hip).pos.copy()
                )

                # convert to world coordinates
                robot_hip_world = (
                    base_pos
                    + base_rotation.apply(robot_hip_local)
                )

                # reconstruct wrist/wheel target
                human_data[human_wrist][0] = (
                    robot_hip_world + arm_vector
                )
        self.scaled_human_data = human_data

        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
        
        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

        # for name in [
        #     "left_shoulder",
        #     "right_shoulder",
        #     "left_elbow",
        #     "right_elbow",
        #     "left_wrist",
        #     "right_wrist",
        # ]:
        #     print(name, human_data[name][0])

        # for body_name in [
        #     "FL_hip", "FR_hip",
        #     "FL_calf", "FR_calf",
        #     "FL_foot", "FR_foot",
        # ]:
        #     body_id = mj.mj_name2id(
        #         self.model,
        #         mj.mjtObj.mjOBJ_BODY,
        #         body_name
        #     )
        #     print(body_name, self.configuration.data.xpos[body_id])

        # stop = input()
            
    def retarget(self, human_data, offset_to_ground=False):
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)

        # self._debug_frame = getattr(self, "_debug_frame", 0) + 1

        # print(f"\n========== FRAME {self._debug_frame} TARGETS ==========")

        # # table 2 中 pelvis 对应的 base_link target
        # base_task = self.human_body_to_task2["pelvis"]
        # base_target = base_task.transform_target_to_world

        # base_target_pos = base_target.translation()
        # base_target_rot = base_target.rotation().as_matrix()

        # # world -> target base frame
        # world_to_base_rot = base_target_rot.T

        # if self.tgt_robot == "unitree_go2w":
        #     target_frames = {
        #         "FL_hip",
        #         "FL_calf",
        #         "FL_wheel_link",
        #         "FR_hip",
        #         "FR_calf",
        #         "FR_wheel_link",
        #     }
        # else:
        #     target_frames = {
        #         "FL_hip",
        #         "FL_calf",
        #         "FL_foot",
        #         "FR_hip",
        #         "FR_calf",
        #         "FR_foot",
        #     }

        # for human_name, task in self.human_body_to_task2.items():
        #     if task.frame_name not in target_frames:
        #         continue

        #     target_pose = task.transform_target_to_world
        #     target_world = target_pose.translation()

        #     target_in_base = world_to_base_rot @ (
        #         target_world - base_target_pos
        #     )

        #     print(
        #         f"{human_name:15s} -> {task.frame_name:15s}"
        #     )
        #     print(
        #         "  target world:",
        #         np.round(target_world, 4),
        #     )
        #     print(
        #         "  target base :",
        #         np.round(target_in_base, 4),
        #         "distance:",
        #         round(np.linalg.norm(target_in_base), 4),
        #     )
        # # stop = input()
        
        if self.use_ik_match_table1:
            # Solve the IK problem
            curr_error = self.error1()
            dt = self.configuration.model.opt.timestep
            vel1 = mink.solve_ik(
                self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel1 = mink.solve_ik(
                    self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter += 1

        if self.use_ik_match_table2:
            curr_error = self.error2()
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(
                self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                # Solve the IK problem with the second task
                dt = self.configuration.model.opt.timestep
                vel2 = mink.solve_ik(
                    self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel2, dt)
                
                next_error = self.error2()
                num_iter += 1
                
            
        return self.configuration.data.qpos.copy()


    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            if body_name == "spine3" and self.tgt_robot in ["unitree_go2", "unitree_go2w"]:
                continue
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat
            
            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
            
            offset_human_data[body_name][0] = pos + global_pos_offset
           
        return offset_human_data
            
    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data
