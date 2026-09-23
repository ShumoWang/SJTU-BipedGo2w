"""Independent acceptance gates. Export remains a candidate if dynamics fails."""
from pathlib import Path
import json,hashlib
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
from .geometry import pair_distance,object_outward_normal

def run(cfg,out):
 out=Path(out);z=dict(np.load(out/'reference.npz'));m=mujoco.MjModel.from_xml_path(str(out/'scene.xml'));d=mujoco.MjData(m);q=z['qpos'];obj=m.geom('payload_box').id;gids=[np.where(m.geom_bodyid==m.body(l+'_wheel_link').id)[0][0] for l in ['FL','FR']];distances=[];normal_cos=[];manifold_present=[];penetration=np.zeros(3);bottom=[]
 for k in range(len(q)):
  d.qpos[:23]=q[k];d.qpos[23:26]=z['object_pos'][k];d.qpos[26:]=z['object_quat'][k];mujoco.mj_forward(m,d)
  row=[]
  for g in gids:
   row.append(pair_distance(m,d,g,obj))
  distances.append(row)
  rr=R.from_quat(z['object_quat'][k,[1,2,3,0]]);nc=[];mp=[]
  for i,g in enumerate(gids):
   normals,present=object_outward_normal(m,d,g,obj);nc.append(max([rr.inv().apply(n)[1]*(1 if i==0 else -1) for n in normals],default=-1));mp.append(present)
  normal_cos.append(nc);manifold_present.append(mp)
  for c in d.contact:
   a,b=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2];i=0 if obj in [c.geom1,c.geom2] else 1 if a and b else 2;penetration[i]=max(penetration[i],-c.dist)
  rot=R.from_quat(z['object_quat'][k,[1,2,3,0]]);bottom.append(z['object_pos'][k,2]-np.abs(rot.as_matrix()[2])@z['object_half_size'])
 st=z['source_time'];closed=(st>=34/30)&(st<=153/30);distances=np.array(distances);bottom=np.array(bottom);ids=np.array([i for i in range(16) if i%4!=3]);lim=m.jnt_range[1:17];jerr=float(max(0,np.max(lim[ids,0]-q[:,7+ids]),np.max(q[:,7+ids]-lim[ids,1])));normerr=float(np.max(abs(np.linalg.norm(z['object_quat'],axis=1)-1)))
 # World angular velocity: central differencing on SO(3), not quaternion components.
 rot=R.from_quat(z['object_quat'][:,[1,2,3,0]]);dt=1/cfg['output_fps'];omega=np.zeros((len(q),3));omega[1:-1]=(rot[2:]*rot[:-2].inv()).as_rotvec()/(2*dt);omega[0]=(rot[1]*rot[0].inv()).as_rotvec()/dt;omega[-1]=(rot[-1]*rot[-2].inv()).as_rotvec()/dt
 z['object_ang_vel_world']=omega;np.savez(out/'reference.npz',**z)
 normal_cos=np.array(normal_cos);manifold_present=np.array(manifold_present)
 gates=dict(opposing_side_normals=bool(normal_cos[closed].min()>.8),finite=bool(all(np.isfinite(z[k]).all() for k in ['qpos','qvel','qacc','object_pos','object_quat'])),joint_limits=jerr<1e-5,object_quaternion=normerr<1e-8,nonpenetration=bool(np.max(penetration)<.002),bilateral_contact_gap=bool(np.abs(distances[closed]).max()<.003),ground_start_end=bool(np.max(abs(bottom[[0,-1]]))<1e-8),ground_not_penetrated=bool(bottom.min()>-1e-8),rolling=bool(z['wheel_metrics'][:,:,2].max()<.003))
 rt=json.loads((out/'retarget_report.json').read_text());gates['rotation_prior_improved']=rt['rotation_error_after_carry_rms_deg']<rt['rotation_error_before_carry_rms_deg'];kinematic=all(gates.values());physics_path=out/'physics_report.json';physics=json.loads(physics_path.read_text()) if physics_path.exists() else None
 reference_hash=hashlib.sha256((out/'reference.npz').read_bytes()).hexdigest()
 if physics is not None and physics.get('reference_sha256')!=reference_hash:physics=None
 # Current stress controller is diagnostic, not certification of all payload contacts.
 report=dict(status='KINEMATIC_REFERENCE_PASS_DYNAMICS_UNVALIDATED' if kinematic else 'KINEMATIC_REFERENCE_REJECTED',gates=gates,kinematic_reference_ready=kinematic,training_ready=False,payload_manipulation_validated=False,physics_test=physics,box_robot_penetration_max_m=float(penetration[0]),self_penetration_max_m=float(penetration[1]),floor_penetration_max_m=float(penetration[2]),joint_limit_violation_rad=jerr,closed_contact_gap_max_m=float(np.abs(distances[closed]).max()),side_normal_cos_min=float(normal_cos[closed].min()),closed_narrow_phase_contact_fraction=manifold_present[closed].mean(0).tolist(),object_ground_start_end_m=bottom[[0,-1]].tolist(),no_slip_max_m_s=float(z['wheel_metrics'][:,:,2].max()),object_angular_speed_max_rad_s=float(np.linalg.norm(omega,axis=1).max()),object_linear_speed_max_m_s=float(np.linalg.norm(z['object_lin_vel'],axis=1).max()),reference_sha256=hashlib.sha256((out/'reference.npz').read_bytes()).hexdigest())
 (out/'acceptance.json').write_text(json.dumps(report,indent=2));np.savez(out/'validation.npz',wheel_box_distance=distances,box_bottom=bottom,side_normal_cos=normal_cos,narrow_phase_contact=manifold_present)
 metadata=dict(schema_version='pedhoi-1',frame='world Z-up, meters, seconds',robot_qpos='base xyz, quaternion wxyz, 16 joints in joint_names order',robot_qvel='MuJoCo nv22: linear world, angular body-local, joint velocities',object_quaternion='wxyz',object_ang_vel='world',legacy_root_rot='xyzw; prefer robot_root_quat_wxyz',source_sequence=cfg['sequence'],kinematic_reference_ready=kinematic,training_ready=False,payload_manipulation_validated=False,mass_is_assumed=True)
 np.savez(out/'episode.npz',**z,robot_root_quat_wxyz=q[:,3:7],object_quat_wxyz=z['object_quat'],metadata_json=json.dumps(metadata),kinematic_reference_ready=kinematic);(out/'schema.json').write_text(json.dumps(metadata,indent=2));return report
if __name__=='__main__':
 import sys
 c=json.loads(Path(sys.argv[1]).read_text());print(json.dumps(run(c,c['output']),indent=2))
