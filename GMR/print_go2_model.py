import mujoco as mj

xml_path = "/home/shumowang/Desktop/mimic_ws/GMR/assets/unitree_go2/go2.xml"
model = mj.MjModel.from_xml_path(xml_path)

def name(object_type, object_id):
    result = mj.mj_id2name(model, object_type, object_id)
    return result if result is not None else "<unnamed>"

print("\n=== JOINTS ===")
for joint_id in range(model.njnt):
    print(
        f"{joint_id:3d}: "
        f"{name(mj.mjtObj.mjOBJ_JOINT, joint_id)} "
        f"qposadr={model.jnt_qposadr[joint_id]} "
        f"dofadr={model.jnt_dofadr[joint_id]}"
    )

print("\n=== BODIES ===")
for body_id in range(model.nbody):
    print(
        f"{body_id:3d}: "
        f"{name(mj.mjtObj.mjOBJ_BODY, body_id)} "
        f"parent_id={model.body_parentid[body_id]}"
    )

print("\n=== ACTUATORS ===")
for actuator_id in range(model.nu):
    print(
        f"{actuator_id:3d}: "
        f"{name(mj.mjtObj.mjOBJ_ACTUATOR, actuator_id)}"
    )

print("\n=== DOFS ===")
for dof_id in range(model.nv):
    joint_id = model.dof_jntid[dof_id]
    print(
        f"{dof_id:3d}: "
        f"{name(mj.mjtObj.mjOBJ_JOINT, joint_id)}"
    )


print("\n=== JOINT LIMITS ===")
for joint_id in range(model.njnt):
    joint_name = mj.mj_id2name(
        model,
        mj.mjtObj.mjOBJ_JOINT,
        joint_id,
    )

    if model.jnt_limited[joint_id]:
        lower, upper = model.jnt_range[joint_id]
        print(
            f"{joint_id}: {joint_name}: "
            f"{lower:.4f} to {upper:.4f}"
        )