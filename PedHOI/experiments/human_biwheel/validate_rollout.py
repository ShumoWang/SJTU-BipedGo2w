from pathlib import Path
import json,sys
import numpy as np,mujoco
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R
P=Path(__file__).resolve().parent
m=mujoco.MjModel.from_xml_path('/home/robotennis2025/SJTU-BipedGo2w/GMR/assets/unitree_go2w/go2w.xml');d=mujoco.MjData(m);ref=mujoco.MjData(m)
a=np.load(P/'balance_tracking_rollout.npz');r=a['records'];z=np.load(P/'reference.npz');interp=PchipInterpolator(z['time'],z['qpos']);rows=[]
for row in r:
 t=row[0];d.qpos[:]=row[2:25];d.qvel[:]=row[25:47];mujoco.mj_forward(m,d);qref=interp(min(t+.002,z['time'][-1]));qref[3:7]/=np.linalg.norm(qref[3:7]);ref.qpos[:]=qref;mujoco.mj_forward(m,ref)
 vals=[]
 for name in ['RL','RR']:
  j=m.joint(name+'_wheel_joint').id;b=m.body(name+'_wheel_link').id;axis=d.xaxis[j].copy();lat=axis.copy();lat[2]=0;lat/=np.linalg.norm(lat);normal=np.array([0,0,1.]);rad=normal-axis[2]*axis;rad=-.086*rad/np.linalg.norm(rad);point=d.xanchor[j]+rad;J=np.zeros((3,m.nv));mujoco.mj_jac(m,d,J,None,point,b);vc=J@d.qvel
  contacts=[c for c in d.contact if (m.geom_bodyid[c.geom1]==b and m.geom_bodyid[c.geom2]==0) or (m.geom_bodyid[c.geom2]==b and m.geom_bodyid[c.geom1]==0)]
  vals += [float(vc@lat),float(np.linalg.norm(vc[:2])),float(point[2]),float(len(contacts)>0)]
 for name in ['FL_wheel_link','FR_wheel_link']:vals.append(float(np.linalg.norm(d.body(name).xpos-ref.body(name).xpos)))
 vals.append(float((R.from_quat(d.qpos[[4,5,6,3]])*R.from_quat(qref[[4,5,6,3]]).inv()).magnitude()))
 rows.append(vals)
rows=np.array(rows);carry=(r[:,0]>=6.8)&(r[:,0]<=17.47)
def stats(x):return dict(rms=float(np.sqrt(np.mean(x*x))),p95_abs=float(np.percentile(np.abs(x),95)),max_abs=float(np.max(np.abs(x))))
report=dict(rollout='balance_tracking',completed=bool(json.loads((P/'balance_tracking_report.json').read_text())['completed']),root_position_error_m=stats(r[:,1]),base_orientation_error_deg=stats(np.degrees(rows[:,-1])),actual_rear_lateral_slip_m_s=stats(rows[:,[0,4]]),actual_carry_rear_lateral_slip_m_s=stats(rows[carry][:,[0,4]]),actual_rear_planar_contact_speed_m_s=stats(rows[:,[1,5]]),rear_wheel_contact_fraction=rows[:,[3,7]].mean(axis=0).tolist(),fore_wheel_tracking_error_m=stats(rows[:,[8,9]]),base_height_min_m=float(r[:,4].min()),scope='unloaded original-MJCF rollout, material point evaluated on ideal wheel; mesh/contact discretization may differ; no object grasp',acceptance_thresholds=dict(root_error_max_m=.2,root_error_rms_m=.12,orientation_error_max_deg=20,carry_lateral_slip_rms_m_s=.03,each_rear_contact_fraction=.95))
report['unloaded_rollout_pass']=bool(report['completed'] and report['root_position_error_m']['max_abs']<.2 and report['root_position_error_m']['rms']<.12 and report['base_orientation_error_deg']['max_abs']<20 and report['actual_carry_rear_lateral_slip_m_s']['rms']<.03 and min(report['rear_wheel_contact_fraction'])>.95)
np.savez(P/'rollout_metrics.npz',time=r[:,0],metrics=rows)
(P/'rollout_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
