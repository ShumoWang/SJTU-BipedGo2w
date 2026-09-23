"""Torque-only whole-body QP replay. Run with gmr environment (proxqp).
Variables: generalized acceleration, 16 motor torques, two 3D contact forces.
"""
from pathlib import Path
import json,sys,os
import numpy as np,mujoco
from scipy.interpolate import PchipInterpolator
from qpsolvers import solve_qp
P=Path(__file__).resolve().parent
MODEL=os.environ.get('PED_MODEL','/home/robotennis2025/SJTU-BipedGo2w/GMR/assets/unitree_go2w/go2w.xml')
TAG=os.environ.get('PED_TAG','wbc')
m=mujoco.MjModel.from_xml_path(MODEL);m.opt.timestep=.002;m.opt.iterations=60
z=np.load(P/'reference.npz');tt=z['time'];qr=PchipInterpolator(tt,z['qpos']);vr=PchipInterpolator(tt,z['qvel']);ar=PchipInterpolator(tt,z['qacc'])
nv,nu=m.nv,m.nu;S=np.zeros((nv,nu));acts=[]
for k in range(nu):j=m.actuator_trnid[k,0];vi=m.jnt_dofadr[j];S[vi,k]=m.actuator_gear[k,0];acts.append(vi)
d=mujoco.MjData(m);tmp=mujoco.MjData(m);d.qpos[:]=qr(0);d.qvel[:]=vr(0);mujoco.mj_forward(m,d)
normal=np.array([0.,0.,1.]);rad=.086

def contact(data):
 Js=[];points=[]
 for name in ['RL','RR']:
  j=m.joint(name+'_wheel_joint').id;b=m.body(name+'_wheel_link').id;axis=data.xaxis[j]
  r=normal-axis[2]*axis;r=-rad*r/np.linalg.norm(r);p=data.xanchor[j]+r;J=np.zeros((3,nv));mujoco.mj_jac(m,data,J,None,p,b);Js.append(J);points.append(p)
 return np.concatenate(Js),np.array(points)
kp=np.r_[35,35,150,180,180,180,np.full(16,70.)];kd=2*np.sqrt(kp)
weights=np.r_[8,8,15,25,25,25,np.full(16,1.)]
for j in range(1,m.njnt):
 if 'wheel' in m.joint(j).name:kp[m.jnt_dofadr[j]]=0;kd[m.jnt_dofadr[j]]=5;weights[m.jnt_dofadr[j]]=.2
# friction cone inscribed square
G=[]
for i in range(2):
 for xy in range(2):
  for sign in [-1,1]:
   row=np.zeros(nv+nu+6);row[nv+nu+i*3+xy]=sign;row[nv+nu+i*3+2]=-.8/np.sqrt(2);G.append(row)
G=np.array(G);lb=np.r_[np.full(nv,-200),m.actuator_ctrlrange[:,0],[-1000,-1000,0]*2];ub=np.r_[np.full(nv,200),m.actuator_ctrlrange[:,1],[1000,1000,1000]*2]
H=np.diag(np.r_[weights,np.full(nu,1e-4),np.full(6,1e-6)])
dx=np.zeros(nv);records=[];controls=[];failed=None;solver_fail=0;last=None
DT=.01
for k in range(int(tt[-1]/DT)):
 t=k*DT;qref=qr(t);qref[3:7]/=np.linalg.norm(qref[3:7]);vref=vr(t);aref=ar(t)
 mujoco.mj_forward(m,d);mujoco.mj_differentiatePos(m,dx,1,d.qpos,qref)
 desired=aref+kp*dx+kd*(vref-d.qvel)
 J,points=contact(d);tmp.qpos[:]=d.qpos;mujoco.mj_integratePos(m,tmp.qpos,d.qvel,1e-4);tmp.qvel[:]=d.qvel;mujoco.mj_forward(m,tmp);Jnext,_=contact(tmp);jdv=(Jnext-J)@d.qvel/1e-4
 target=-jdv-15*(J@d.qvel);target[[2,5]]-=100*points[:,2]
 M=np.zeros((nv,nv));mujoco.mj_fullM(m,M,d.qM)
 E=np.vstack([np.c_[M,-S,-J.T],np.c_[J,np.zeros((6,nu+6))]])
 b=np.r_[-d.qfrc_bias+d.qfrc_passive,target]
 g=np.r_[-weights*desired,np.zeros(nu+6)]
 sol=solve_qp(H,g,G,np.zeros(8),E,b,lb,ub,solver='proxqp',eps_abs=1e-5,max_iter=100)
 if sol is None or not np.isfinite(sol).all():solver_fail+=1;failed=t;break
 d.ctrl[:]=sol[nv:nv+nu];controls.append(d.ctrl.copy())
 for _ in range(5):mujoco.mj_step(m,d)
 error=np.linalg.norm(d.qpos[:3]-qref[:3]);records.append(np.r_[t,error,d.qpos.copy(),d.qvel.copy()])
 if k%500==0:print('WBC',t,'error',round(error,4),flush=True)
 if d.qpos[2]<.25 or error>.5 or not np.isfinite(d.qpos).all():failed=t;break
np.savez(P/(TAG+'_rollout.npz'),records=records,ctrl=controls)
report=dict(controller='contact-constrained whole-body QP, 100 Hz / 500 Hz physics',model=MODEL,torque_only=True,root_wrench=False,position_resets=False,payload=False,completed=failed is None,planned_s=float(tt[-1]),actual_s=len(records)*DT,stop_time_s=failed,solver_failures=solver_fail,root_error_rms_m=float(np.sqrt(np.mean(np.array(records)[:,1]**2))) if records else None,root_error_max_m=float(np.max(np.array(records)[:,1])) if records else None)
(P/(TAG+'_report.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
