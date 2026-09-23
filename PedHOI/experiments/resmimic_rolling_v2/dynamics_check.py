"""Unloaded two-point-contact inverse dynamics screen + torque-only PD rollout.
LP uses conservative square inside friction cone. No external stabilizing wrench.
Rollout failure is controller/model-specific, not proof no controller can succeed.
"""
from pathlib import Path
import json
import numpy as np
import mujoco
from scipy.optimize import linprog
from build_reference import MODEL,RADIUS
P=Path(__file__).resolve().parent
m=mujoco.MjModel.from_xml_path(MODEL);d=mujoco.MjData(m)
a=np.load(P/'reference.npz');q=a['qpos'];v=a['qvel'];acc=a['qacc'];times=a['time']
S=np.zeros((m.nv,m.nu))
for k in range(m.nu):S[m.jnt_dofadr[m.actuator_trnid[k,0]],k]=m.actuator_gear[k,0]
results=[];torques=[]
for f in range(0,len(q),6):
 d.qpos[:]=q[f];d.qvel[:]=v[f];mujoco.mj_forward(m,d)
 M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,M,d.qM)
 rhs=M@acc[f]+d.qfrc_bias-d.qfrc_passive
 Js=[]
 for name in ['RL','RR']:
  j=m.joint(name+'_wheel_joint').id;b=m.body(name+'_wheel_link').id
  axis=d.xaxis[j];normal=np.array([0.,0.,1.]);r=normal-axis[2]*axis;r=-RADIUS*r/np.linalg.norm(r)
  J=np.zeros((3,m.nv));mujoco.mj_jac(m,d,J,None,d.xanchor[j]+r,b);Js.append(J)
 A=np.c_[S,Js[0].T,Js[1].T];ineq=[]
 for i in range(2):
  for xy in range(2):
   for sign in [-1,1]:
    row=np.zeros(22);row[16+3*i+xy]=sign;row[18+3*i]=-.8/np.sqrt(2);ineq.append(row)
 bounds=[tuple(x) for x in m.actuator_ctrlrange]+[(None,None),(None,None),(0,None)]*2
 # slack on all equations, minimize L1 residual. Distinguish base force/torque.
 E=np.c_[A,np.eye(m.nv),-np.eye(m.nv)]
 res=linprog(np.r_[np.zeros(22),np.ones(2*m.nv)],A_ub=np.c_[ineq,np.zeros((8,2*m.nv))],b_ub=np.zeros(8),A_eq=E,b_eq=rhs,bounds=bounds+[(0,None)]*(2*m.nv),method='highs')
 residual=A@res.x[:22]-rhs if res.success else np.full(m.nv,np.nan)
 results.append([times[f],np.max(np.abs(residual)),np.linalg.norm(residual[:3]),np.linalg.norm(residual[3:6])]);torques.append(res.x[:16] if res.success else np.zeros(16))
rows=np.array(results);np.savez(P/'inverse_dynamics.npz',metrics=rows,torques=torques,sample_frames=np.arange(0,len(q),6))
# Torque-only replay initialized in transport at t=3.5. No qpos overwrite afterwards.
start=int(3.5*120);d=mujoco.MjData(m);d.qpos[:]=q[start];d.qvel[:]=v[start];mujoco.mj_forward(m,d)
records=[];failed=None
for step in range(1500):
 t=times[start]+step*m.opt.timestep;f=min(int(round(t*120)),len(q)-1)
 for k in range(m.nu):
  j=m.actuator_trnid[k,0];qi=m.jnt_qposadr[j];vi=m.jnt_dofadr[j];wheel='wheel' in m.joint(j).name
  tau=(0 if wheel else 50)*(q[f,qi]-d.qpos[qi])+(2 if wheel else 2.5)*(v[f,vi]-d.qvel[vi])
  d.ctrl[k]=np.clip(tau,*m.actuator_ctrlrange[k])
 mujoco.mj_step(m,d)
 error=np.linalg.norm(d.qpos[:3]-q[f,:3]);records.append(np.r_[t,error,d.qpos.copy()])
 if d.qpos[2]<.25 or error>.5 or not np.isfinite(d.qpos).all():failed=t;break
np.savez(P/'pd_rollout.npz',records=records)
report=dict(assumptions=['unloaded robot','two ideal point contacts, rear wheels only','mu=0.8 with inscribed square friction cone','model actuator ctrlrange bounds','passive forces from MuJoCo included; dry joint friction not separately modeled'],inverse_dynamics_samples=len(rows),feasible_fraction_under_1e_4=float(np.mean(rows[:,1]<1e-4)),transport_feasible_fraction_under_1e_4=float(np.mean(rows[(rows[:,0]>=3.33)&(rows[:,0]<=4.67),1]<1e-4)),maximum_generalized_equation_residual=float(rows[:,1].max()),pd_rollout=dict(controller='leg position/velocity PD + wheel velocity PD, no root wrench, no feedforward or balance controller',start_s=3.5,simulated_s=len(records)*m.opt.timestep,failure_time_s=failed,last_root_position_error_m=float(records[-1][1]),interpretation='controller stress test only, not a proof of impossibility'),status='NOT dynamically validated')
(P/'dynamics_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
