"""Direct Isaac replay of the exact PedHOI reference, with no physics integration."""
from pathlib import Path
import hashlib
import json
import time
import shutil
import xml.etree.ElementTree as ET
import numpy as np


def run_pedhoi(args, app):
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.asset.importer.mjcf")
    import torch
    import imageio.v2 as imageio
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.math import quat_apply

    path = Path(args.motion_file).resolve()
    with np.load(path, allow_pickle=False) as data:
        z = {k: data[k] for k in data.files}
    fps = float(z['fps']); q = z['qpos']; n = len(q)
    if not np.isfinite(fps) or fps <= 0 or n == 0:
        raise ValueError('Expected a nonempty trajectory with positive FPS')
    if args.playback_speed <= 0 or args.render_fps <= 0:
        raise ValueError('Playback speed and render FPS must be positive')
    if q.shape != (n, 23) or len(z['joint_names']) != 16:
        raise ValueError('Expected the named 16-joint Go2W reference')
    out = Path(args.video_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.robot_xml).resolve() if args.robot_xml else Path(__file__).resolve().parents[2] / 'GMR/assets/unitree_go2w/go2w.xml'
    # Same robot geometry, joints and body frames as MuJoCo video. Only remove
    # the world floor; Isaac supplies its own floor. Copy mesh files locally:
    # the importer writes temporary files beside them and mishandles absolute meshdir.
    tree = ET.parse(model_path); root = tree.getroot()
    compiler = root.find('compiler')
    mesh_source = (model_path.parent / compiler.get('meshdir', '')).resolve()
    mesh_link = out / 'meshes'
    if mesh_link.is_symlink(): mesh_link.unlink()
    mesh_link.mkdir(exist_ok=True)
    for mesh in root.findall('./asset/mesh'):
        relative = Path(mesh.get('file'))
        target_mesh = mesh_link / relative
        target_mesh.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mesh_source / relative, target_mesh)
    compiler.set('meshdir', 'meshes')
    for geom in list(root.find('worldbody').findall('geom')):
        root.find('worldbody').remove(geom)
    xml = out / 'go2w_replay.xml'; tree.write(xml)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path='/World/Ground', spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(prim_path='/World/Light', spawn=sim_utils.DomeLightCfg(intensity=1600.0))
        robot = ArticulationCfg(
            prim_path='{ENV_REGEX_NS}/Robot', articulation_root_prim_path='/base_link/base_link',
            spawn=sim_utils.MjcfFileCfg(asset_path=str(xml), fix_base=False,
                force_usd_conversion=True, rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={'.*_calf_joint': -1.5, '.*_thigh_joint': 0.8}),
            actuators={'all': ImplicitActuatorCfg(joint_names_expr=['.*'], stiffness=0.0, damping=0.0)},
        )
        box = RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Box', spawn=sim_utils.CuboidCfg(
            size=tuple((2*z['object_half_size']).tolist()),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(z['object_mass'])),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.60, 0.32, 0.12)),
        ))
        camera = None

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/fps, device=args.device))
    cfg = SceneCfg(num_envs=1, env_spacing=2.0)
    if args.video:
        cfg.camera = CameraCfg(prim_path='{ENV_REGEX_NS}/Camera', update_period=0.0,
            height=480, width=640, data_types=['rgb'], spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, horizontal_aperture=27.0, clipping_range=(0.01, 100.0)))
    scene = InteractiveScene(cfg); sim.reset()
    robot, box = scene['robot'], scene['box']
    names = z['joint_names'].tolist()
    ids, matched = robot.find_joints(names, preserve_order=True)
    if list(matched) != names or len(ids) != 16:
        raise ValueError(f'Joint mapping failed: {matched}')
    print('[PedHOI] Isaac joint order:', robot.joint_names, flush=True)
    print('[PedHOI] Source -> Isaac indices:', ids, flush=True)
    print('[PedHOI] Body order:', robot.body_names, flush=True)
    print(f'[PedHOI] {n} frames at {fps} Hz; duration {(n-1)/fps}s; speed {args.playback_speed}x', flush=True)
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=sim.device)
    poses = tensor(q[:, :7]); joints = tensor(q[:, 7:]); objects = tensor(np.c_[z['object_pos'], z['object_quat']])
    indices = np.arange(n) if args.audit_only else np.unique(np.rint(np.arange(0, (n-1)/fps + 1e-8, args.playback_speed/args.render_fps)*fps).astype(int))
    indices = indices[indices < n]
    if args.video and args.video_length > 0:
        indices = indices[:args.video_length]
    writer = imageio.get_writer(out / f'{path.stem}_isaac.mp4', fps=args.render_fps) if args.video else None
    measured_pos=[]; measured_quat=[]; measured_joints=[]; measured_box=[]; measured_bodies=[]
    try:
        for k in indices:
            if not app.is_running(): break
            start=time.monotonic()
            if args.audit_only and k % 512 == 0: print(f'[PedHOI] audit frame {k}/{n}', flush=True)
            pose=poses[k:k+1].clone(); pose[:,:3]+=scene.env_origins
            obj=objects[k:k+1].clone(); obj[:,:3]+=scene.env_origins
            robot.write_root_pose_to_sim(pose)
            robot.write_root_velocity_to_sim(torch.zeros((1,6), device=sim.device))
            robot.write_joint_state_to_sim(joints[k:k+1], torch.zeros((1,16),device=sim.device), joint_ids=ids)
            box.write_root_pose_to_sim(obj)
            box.write_root_velocity_to_sim(torch.zeros((1,6),device=sim.device))
            scene.write_data_to_sim()
            # Match MuJoCo camera: heading+70 degrees, elevation -18, distance 1.75.
            forward=quat_apply(pose[:,3:7],tensor([[0,0,-1]]))
            yaw=torch.atan2(forward[:,1],forward[:,0])+np.deg2rad(70)
            target=pose[:,:3].clone(); target[:,2]=0.4
            offset=torch.stack([1.75*np.cos(np.deg2rad(18))*torch.cos(yaw),1.75*np.cos(np.deg2rad(18))*torch.sin(yaw),torch.ones_like(yaw)*1.75*np.sin(np.deg2rad(18))],dim=-1)
            eye=target+offset
            if not args.audit_only:
                sim.set_camera_view(eye[0].cpu().numpy(),target[0].cpu().numpy())
            if args.video:scene['camera'].set_world_poses_from_view(eye,target)
            if args.audit_only:
                sim.forward()
            else:
                sim.render()  # Intentionally no sim.step(): exact kinematic comparison.
            scene.update(1/fps)
            measured_pos.append(robot.data.root_pos_w[0].cpu().numpy().copy()-scene.env_origins[0].cpu().numpy())
            measured_quat.append(robot.data.root_quat_w[0].cpu().numpy().copy())
            measured_joints.append(robot.data.joint_pos[0,ids].cpu().numpy().copy())
            measured_box.append(box.data.root_state_w[0,:7].cpu().numpy().copy())
            measured_bodies.append(robot.data.body_pos_w[0].cpu().numpy().copy())
            if writer:
                frame=scene['camera'].data.output['rgb'][0,:,:,:3].cpu().numpy()
                writer.append_data(frame)
                if len(measured_pos) in [1, min(46,len(indices)),len(indices)]:imageio.imwrite(out/f'frame_{k:04d}.png',frame)
            if not args.headless and not args.audit_only:time.sleep(max(0,1/args.render_fps-(time.monotonic()-start)))
    finally:
        if writer:writer.close()
    used=indices[:len(measured_pos)]
    np.savez(out/'isaac_readback.npz',indices=used,root_pos=measured_pos,root_quat=measured_quat,joint_pos=measured_joints,object_pose=measured_box,body_pos=measured_bodies,body_names=robot.body_names)
    report=dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),model_xml=str(model_path),source_fps=fps,playback_speed=args.playback_speed,frames=len(used),physics_stepped=False,joint_names=names,joint_indices=ids,
        root_position_error_max_m=float(np.max(np.abs(np.array(measured_pos)-q[used,:3]))),
        joint_error_max_rad=float(np.max(np.abs(np.array(measured_joints)-q[used,7:]))))
    (out/'replay_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
