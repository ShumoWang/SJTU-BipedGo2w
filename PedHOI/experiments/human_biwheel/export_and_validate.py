"""Export inspectable robot reference and a conservative training acceptance gate.
Body/joint order is MJCF order; consumer must map by names, never index blindly.
"""
from pathlib import Path
import sys,pickle,json,hashlib
import numpy as np,mujoco
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R,Slerp
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'resmimic_rolling_v2'))
from build_reference import MODEL,velocities,audit
m=mujoco.MjModel.from_xml_path(MODEL);d=mujoco.MjData(m)
z=dict(np.load(P/'reference.npz'));q=z['qpos'];time=z['time'];fps=float(z['fps']);v=velocities(m,q,1/fps)
ids=np.array([i for i in range(16) if i%4!=3]);lim=m.jnt_range[1:][ids]
assert np.isfinite(q).all();assert np.all(q[:,7+ids]>=lim[:,0]-1e-5);assert np.all(q[:,7+ids]<=lim[:,1]+1e-5)
metrics=audit(m,q,fps);bodies=[];bquat=[];blin=[];bang=[];penetrations=[]
for k in range(len(q)):
 d.qpos[:]=q[k];d.qvel[:]=v[k];mujoco.mj_forward(m,d);bodies.append(d.xpos[1:].copy());bquat.append(d.xquat[1:].copy());lv=[];av=[]
 for b in range(1,m.nbody):
  jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,d.xpos[b],b);av.append(jr@v[k]);lv.append(jp@v[k])
 blin.append(lv);bang.append(av)
 for c in d.contact:
  if m.geom_bodyid[c.geom1] and m.geom_bodyid[c.geom2] and c.dist<-.001:penetrations.append([k,float(c.dist),int(c.geom1),int(c.geom2)])
meta=json.loads((P/'source.json').read_text());dyn=json.loads((P/'dynamics_report.json').read_text());closed=json.loads((P/'balance_tracking_report.json').read_text());rollout=json.loads((P/'rollout_validation.json').read_text())
kinematic=bool(metrics[:,:,2].max()<.003 and np.max(np.abs(metrics[:,:,3]))<1e-5)
force_pass=bool(dyn['feasible_fraction_under_1e_3']==1.)
acceptance=dict(raw_human_input=True,source_sequence=meta['sequence'],no_slip_kinematic_pass=kinematic,no_slip_peak_m_s=float(metrics[:,:,2].max()),no_slip_rms_m_s=float(np.sqrt(np.mean(metrics[:,:,2]**2))),joint_limits_pass=True,reference_self_penetration_over_1mm_count=len(penetrations),unloaded_ideal_force_screen_pass_1e_3=force_pass,strict_force_fraction_1e_4=dyn['feasible_fraction_under_1e_4'],original_model_closed_loop_pass=bool(rollout['unloaded_rollout_pass']),closed_loop_duration_s=closed['actual_s'],tracking_reference_ready=bool(kinematic and force_pass and rollout['unloaded_rollout_pass']),training_ready=False,payload_manipulation_validated=False,status='PASS_UNLOADED_REFERENCE_PIPELINE; OBJECT_MANIPULATION_NOT_VALIDATED',scope='complete retarget and validation pipeline; unloaded tracking reference; HOI contact/load validation remains outside this test',candidate_sha256=hashlib.sha256((P/'reference.npz').read_bytes()).hexdigest())
(P/'acceptance.json').write_text(json.dumps(acceptance,indent=2))
np.savez_compressed(P/'motion_candidate.npz',fps=np.array([fps]),joint_pos=q[:,7:],joint_vel=v[:,6:],body_pos_w=np.array(bodies),body_quat_w=np.array(bquat),body_lin_vel_w=np.array(blin),body_ang_vel_w=np.array(bang),body_names=np.array([m.body(i).name for i in range(1,m.nbody)]),joint_names=np.array([m.joint(i).name for i in range(1,m.njnt)]),training_ready=False,metadata_json=json.dumps(dict(quaternion_order='wxyz',ordering='MuJoCo model order, NOT assumed Isaac Lab order',source=meta['sequence'],status=acceptance['status'])))
with open(P/'reference.pkl','wb') as f:pickle.dump(dict(fps=int(fps),root_pos=q[:,:3],root_rot=q[:,[4,5,6,3]],dof_pos=q[:,7:],local_body_pos=None,link_body_list=None),f)
# A candidate schema can be exported without pretending it is accepted training data.
(P/'schema.json').write_text(json.dumps(dict(reference_npz='qpos base quaternion wxyz; root_rot xyzw; dof_pos ordered by joint_names; fps120',reference_pkl='GMR-compatible generalized-coordinate keys; root_rot xyzw',motion_candidate='body/joint FK and velocities in world coordinates with explicit names; quaternion wxyz; reject unless acceptance training_ready true',phase_timing='source_time indexes raw 30 Hz sequence; output slowed 4x; source phase selection heuristic',units='m, rad, seconds'),indent=2))
print(json.dumps(acceptance,indent=2))
