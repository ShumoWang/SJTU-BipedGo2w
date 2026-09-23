"""Regression tests for data lineage, rolling physics and safe rejection."""
from pathlib import Path
import sys,json
import numpy as np,mujoco
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'resmimic_rolling_v2'))
from build_reference import audit,MODEL
m=mujoco.MjModel.from_xml_path(MODEL)
source=json.loads((P/'source.json').read_text());h=np.load(P/'human_source.npz');z=np.load(P/'reference.npz')
assert source['source_sha256'] and 'SMPL-X' in source['body_model'];assert h['joints_w'].shape==(193,22,3)
# Paired human/object motion includes lift and return to initial height.
height=h['object_pos'][:,2];assert height.max()-height[0]>.5;assert abs(height[-1]-height[0])<.05
met=audit(m,z['qpos'],float(z['fps']));assert met[:,:,2].max()<.003
# Prior sliding reference must be rejected by exactly the same checker.
old=np.load(P.parent/'resmimic_pick_place/go2w_carry_adapted.npz');bad=audit(m,old['qpos'],30);assert bad[100:141,:,2].max()>.3
# Force solution respects real actuator limits and conservative friction bounds.
f=np.load(P/'inverse_dynamics.npz');tau=f['torques'];force=f['contact_forces'].reshape(-1,2,3)
assert np.all(tau>=m.actuator_ctrlrange[:,0]-1e-6);assert np.all(tau<=m.actuator_ctrlrange[:,1]+1e-6)
assert np.all(force[:,:,2]>=-1e-7);assert np.all(np.linalg.norm(force[:,:,:2],axis=2)<=.8*force[:,:,2]+1e-6)
assert f['metrics'][:,1].max()<.001
export=np.load(P/'motion_candidate.npz');assert export['body_pos_w'].shape[1]==len(export['body_names']);assert export['joint_pos'].shape[1]==len(export['joint_names']);assert np.isfinite(export['body_lin_vel_w']).all()
gate=json.loads((P/'acceptance.json').read_text());assert gate['tracking_reference_ready'];assert not gate['training_ready'];assert not export['training_ready']
print('PASS: raw human lineage, pick/place evidence, rolling regression, friction/torque bounds, export schema, rejection gate')
