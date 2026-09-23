"""Sparse trajectory refinement of the unactuated axle moment.
Does not invent external support: adjusts fore-aft COM/axle relation through legs.
"""
from pathlib import Path
import sys,json
import numpy as np,mujoco
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as R
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'resmimic_rolling_v2'))
from build_reference import MODEL,velocities,audit
z=dict(np.load(P/'reference.npz'));q0=z['qpos'];ts=z['time'];fps=float(z['fps']);m=mujoco.MjModel.from_xml_path(MODEL);d=mujoco.MjData(m)
rot=R.from_quat(q0[:,[4,5,6,3]]);fw=rot.apply(np.tile([0.,0.,-1.],(len(q0),1)));lat=np.cross([0,0,1],fw);frames=np.arange(0,len(q0),1);knots=ts[frames]
sx=[]
for q in q0:
 d.qpos[:]=q;mujoco.mj_forward(m,d);sx.append((d.body('RL_wheel_link').xpos-q[:3])@rot[len(sx)].apply([0,0,-1]))
sx=np.array(sx);weight=m.body_mass.sum()*9.81

def candidate(x):
 delta=CubicSpline(knots,x)(ts);q=q0.copy();q[:,:2]-=delta[:,None]*fw[:,:2];xx=sx+delta
 for base,w in [(8,11),(12,15)]:
  dz=.086-(q[:,2]-.1934);c=(xx*xx+dz*dz-.213**2-.2264**2)/(2*.213*.2264);knee=-np.arccos(np.clip(c,-1,1));thigh=np.arctan2(-dz,xx)-np.arctan2(.2264*np.sin(knee),.213+.2264*np.cos(knee))
  dp=thigh+knee-q[:,7+base+1]-q[:,7+base+2];q[:,7+base+1]=thigh;q[:,7+base+2]=knee;q[:,7+w]-=dp-dp[0]
 v=velocities(m,q,1/fps);a=np.gradient(v,1/fps,axis=0)
 return q,v,a,delta

def fun(x):
 q,v,a,_=candidate(x);res=[]
 for k in frames:
  d.qpos[:]=q[k];d.qvel[:]=v[k];mujoco.mj_forward(m,d);M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,M,d.qM);rhs=M@a[k]+d.qfrc_bias-d.qfrc_passive
  center=(d.body('RL_wheel_link').xpos+d.body('RR_wheel_link').xpos)/2;center[2]-=.086
  torque=rot[k].apply(rhs[3:6]);e=(torque-np.cross(center-q[k,:3],rhs[:3]))@lat[k];res.append(e/weight)
 return np.r_[res,.0001*x]
size=len(frames);sp=lil_matrix((2*size,size))
for i in range(size):sp[i,max(0,i-5):min(size,i+6)]=1;sp[size+i,i]=1
before=fun(np.zeros(size))[:size]*weight
fit=least_squares(fun,np.zeros(size),jac_sparsity=sp.tocsr(),bounds=(-.04,.04),max_nfev=25,ftol=1e-8,gtol=1e-9,xtol=1e-8,verbose=1)
q,v,a,delta=candidate(fit.x);after=fun(fit.x)[:size]*weight;metrics=audit(m,q,fps)
np.savez(P/'reference_pre_dense_refinement.npz',**z)
z.update(qpos=q,qvel=v,qacc=a,root_pos=q[:,:3],root_rot=q[:,[4,5,6,3]],dof_pos=q[:,7:],wheel_metrics=metrics,balance_refinement=delta)
np.savez(P/'reference.npz',**z)
report=dict(optimizer_success=bool(fit.success),optimizer_message=fit.message,evals=fit.nfev,axle_moment_before_rms_Nm=float(np.sqrt(np.mean(before**2))),axle_moment_after_rms_Nm=float(np.sqrt(np.mean(after**2))),axle_moment_after_max_Nm=float(np.max(np.abs(after))),max_position_change_m=float(np.max(np.abs(delta))),no_slip_max_m_s=float(metrics[:,:,2].max()))
(P/'balance_refinement.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
