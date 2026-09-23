"""Reduced-order wheel balance LQR + joint torque PD/feedforward.
True free-base dynamics. Controller remains an experiment, no support forces.
"""
from pathlib import Path
import os,json
import numpy as np,mujoco
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R
from scipy.linalg import solve_continuous_are
P=Path(__file__).resolve().parent
m=mujoco.MjModel.from_xml_path('/home/robotennis2025/SJTU-BipedGo2w/GMR/assets/unitree_go2w/go2w.xml');m.opt.timestep=.002;m.opt.iterations=60
z=np.load(P/'reference.npz');tt=z['time'];qr=PchipInterpolator(tt,z['qpos']);vr=PchipInterpolator(tt,z['qvel']);u=np.load(P/'inverse_dynamics.npz');ff=PchipInterpolator(tt[u['sample_frames']],u['torques'])
M,mb,l=4.,15.,.40;g=9.81
A=np.array([[0,1,0,0],[0,0,-mb*g/M,0],[0,0,0,1],[0,0,(M+mb)*g/(M*l),0]])
B=np.array([[0],[1/M],[0],[-1/(M*l)]])
Q=np.diag([5,8,180,8]);RR=np.array([[.1]]);pc=solve_continuous_are(A,B,Q,RR);K=np.linalg.solve(RR,B.T@pc)
d=mujoco.MjData(m);d.qpos[:]=qr(0);d.qvel[:]=vr(0);mujoco.mj_forward(m,d)
records=[];controls=[];failed=None;DX=np.zeros(m.nv)
for k in range(int(tt[-1]/.002)):
 t=k*.002;qref=qr(t);qref[3:7]/=np.linalg.norm(qref[3:7]);vref=vr(t);Rref=R.from_quat(qref[[4,5,6,3]]);Ra=R.from_quat(d.qpos[[4,5,6,3]])
 forward=Rref.apply([0,0,-1]);forward[2]=0;forward/=np.linalg.norm(forward);lateral=np.cross([0,0,1],forward)
 tilt=np.arctan2(Ra.apply([1,0,0])@forward,Ra.apply([1,0,0])[2]);omega=Ra.apply(d.qvel[3:6])@lateral
 state=np.array([(d.qpos[:3]-qref[:3])@forward,(d.qvel[:3]-vref[:3])@forward,tilt,omega])
 Kuse=K.copy();Kuse[0,0]*=8;Kuse[0,1]*=4;Kuse[0,2]*=float(os.environ.get('PED_KP','4'));Kuse[0,3]*=float(os.environ.get('PED_KD','2'))
 force=float((-Kuse@state).item());wheel_common=force*.086/2
 mujoco.mj_differentiatePos(m,DX,1,qref,d.qpos)
 yawerror=(Ra*Rref.inv()).as_rotvec()[2];yawvel=Ra.apply(d.qvel[3:6])[2]-Rref.apply(vref[3:6])[2]
 lateral_error=(d.qpos[:3]-qref[:3])@lateral
 drive_sign=np.sign(vref[:3]@forward) if abs(vref[:3]@forward)>.02 else 0.
 yawerror+=np.clip(1.5*lateral_error*drive_sign,-.4,.4)
 yawtorque=-40*yawerror-8*yawvel
 baseff=ff(t)
 for i in range(m.nu):
  j=m.actuator_trnid[i,0];qi=m.jnt_qposadr[j];vi=m.jnt_dofadr[j];name=m.joint(j).name
  if name.startswith('RL_wheel'):tau=wheel_common-yawtorque*.086/(2*.142)
  elif name.startswith('RR_wheel'):tau=wheel_common+yawtorque*.086/(2*.142)
  elif 'wheel' in name:tau=baseff[i]-1.5*(d.qvel[vi]-vref[vi])
  else:tau=baseff[i]+100*(qref[qi]-d.qpos[qi])+5*(vref[vi]-d.qvel[vi])
  d.ctrl[i]=np.clip(tau,*m.actuator_ctrlrange[i])
 mujoco.mj_step(m,d)
 err=np.linalg.norm(d.qpos[:3]-qref[:3])
 if k%5==0:records.append(np.r_[t,err,d.qpos.copy(),d.qvel.copy()]);controls.append(d.ctrl.copy())
 if d.qpos[2]<.25 or err>.5 or not np.isfinite(d.qpos).all():failed=t;break
np.savez(P/(os.environ.get('PED_TAG','balance')+'_rollout.npz'),records=records,ctrl=controls)
report=dict(controller='reduced-order cart-pole LQR wheel balancing + bounded joint PD/feedforward',completed=failed is None,actual_s=(k+1)*.002,planned_s=float(tt[-1]),stop_time_s=failed,error_rms_m=float(np.sqrt(np.mean(np.array(records)[:,1]**2))),error_max_m=float(np.max(np.array(records)[:,1])),root_wrench=False,position_reset=False,payload=False,K=Kuse.tolist())
(P/(os.environ.get('PED_TAG','balance')+'_report.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
