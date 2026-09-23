"""Time-varying linear feedback on the original MuJoCo plant; no root wrench.
Finite-difference contact linearization, inverse-dynamics torque feedforward.
No resetting qpos after initialization. A failed rollout remains a failed rollout.
"""
from pathlib import Path
import json,time,os
import numpy as np,mujoco
from scipy.interpolate import PchipInterpolator
from scipy.linalg import solve_discrete_are
P=Path(__file__).resolve().parent
MODEL=os.environ.get('PED_MODEL','/home/robotennis2025/SJTU-BipedGo2w/GMR/assets/unitree_go2w/go2w.xml')
TAG=os.environ.get('PED_TAG','closed_loop')
m=mujoco.MjModel.from_xml_path(MODEL);m.opt.timestep=.002;m.opt.iterations=60
z=np.load(P/'reference.npz');times=z['time'];u=np.load(P/'inverse_dynamics.npz');ff=PchipInterpolator(times[u['sample_frames']],u['torques'],axis=0)
Qref=PchipInterpolator(times,z['qpos'],axis=0);Vref=PchipInterpolator(times,z['qvel'],axis=0)
DT=.01;steps=int(times[-1]/DT);knots=np.arange(0,steps,10);AA=[];BB=[]
for count,k in enumerate(knots):
 d=mujoco.MjData(m);d.qpos[:]=Qref(k*DT);d.qpos[3:7]/=np.linalg.norm(d.qpos[3:7]);d.qpos[2]+=float(os.environ.get("PED_CONTACT_OFFSET","0.0002"))
 d.qvel[:]=Vref(k*DT);d.ctrl[:]=ff(k*DT);mujoco.mj_forward(m,d)
 A=np.zeros((44,44));B=np.zeros((44,16));mujoco.mjd_transitionFD(m,d,1e-5,0,A,B,None,None)
 a=np.eye(44);b=np.zeros((44,16))
 for _ in range(5):b=A@b+B;a=A@a
 AA.append(a);BB.append(b)
 if count%50==0:print('linearized',count,'/',len(knots),flush=True)
AA=np.array(AA);BB=np.array(BB)
pos=np.r_[300,300,500,700,700,700,np.full(16,35.)];vel=np.r_[40,40,60,40,40,40,np.full(16,.8)]
for j in range(1,m.njnt):
 if 'wheel' in m.joint(j).name:pos[m.jnt_dofadr[j]]=.02;vel[m.jnt_dofadr[j]]=1.
Q=np.diag(np.r_[pos,vel]);RR=np.eye(16)*.06;Pcost=Q.copy()*100;Ks=np.zeros((steps,16,44))
for k in range(steps-1,-1,-1):
 A=AA[k//10];B=BB[k//10];K=np.linalg.solve(RR+B.T@Pcost@B,B.T@Pcost@A);Ks[k]=K
 Pcost=Q+A.T@Pcost@(A-B@K);Pcost=(Pcost+Pcost.T)/2
np.savez(P/'feedback_gains.npz',gains=Ks,dt=DT)
d=mujoco.MjData(m);d.qpos[:]=Qref(0);d.qvel[:]=Vref(0);mujoco.mj_forward(m,d)
records=[];ctrl=[];slips=[];failed=None;dx=np.zeros(m.nv)
for k in range(steps):
 t=k*DT;qr=Qref(t);qr[3:7]/=np.linalg.norm(qr[3:7]);vr=Vref(t)
 mujoco.mj_differentiatePos(m,dx,1,qr,d.qpos)
 raw=ff(t)-Ks[k]@np.r_[dx,d.qvel-vr]
 d.ctrl[:]=np.clip(raw,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);ctrl.append(d.ctrl.copy())
 for _ in range(5):mujoco.mj_step(m,d)
 err=np.linalg.norm(d.qpos[:3]-qr[:3]);records.append(np.r_[t,err,d.qpos.copy(),d.qvel.copy()])
 if d.qpos[2]<.25 or err>.5 or not np.isfinite(d.qpos).all():failed=t;break
np.savez(P/(TAG+'_rollout.npz'),records=records,ctrl=ctrl)
report=dict(controller='TVLQR + bounded inverse-dynamics feedforward, 100 Hz feedback, 500 Hz physics',plant=MODEL,root_wrench=False,position_resets=False,payload=False,linearization_contact_height_offset_m=float(os.environ.get("PED_CONTACT_OFFSET","0.0002")),planned_duration_s=float(times[-1]),actual_duration_s=len(records)*DT,completed=failed is None,stop_time_s=failed,stop_reason=None if failed is None else 'root error >0.5m or base height <0.25m or nonfinite state',root_position_error_rms_m=float(np.sqrt(np.mean(np.array(records)[:,1]**2))),root_position_error_max_m=float(np.max(np.array(records)[:,1])),scope='unloaded controller validation only; not evidence of physical grasp')
(P/(TAG+'_report.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
