"""Differential-drive reference with grounded parallel rear wheels.
Preserves prior front-leg take/place articulation; jointly changes rear pose,
root position/heading, and wheel phase. This is not a policy or dynamics proof.
"""
from pathlib import Path
import json,pickle
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline,PchipInterpolator
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation as R
OUT=Path(__file__).resolve().parent
MODEL='/home/robotennis2025/SJTU-BipedGo2w/GMR/assets/unitree_go2w/go2w.xml'
RADIUS=.086

def velocities(m,q,dt):
 v=np.zeros((len(q),m.nv))
 for i in range(len(q)):
  a=max(i-1,0);b=min(i+1,len(q)-1)
  mujoco.mj_differentiatePos(m,v[i],(b-a)*dt,q[a],q[b])
 return v

def audit(m,q,fps):
 d=mujoco.MjData(m);v=velocities(m,q,1/fps);out=[]
 for k in range(len(q)):
  d.qpos[:]=q[k];d.qvel[:]=v[k];mujoco.mj_forward(m,d);row=[]
  forward=R.from_quat(q[k,[4,5,6,3]]).apply([0,0,-1]);forward[2]=0;forward/=np.linalg.norm(forward)
  for name in ['RL','RR']:
   j=m.joint(name+'_wheel_joint').id;b=m.body(name+'_wheel_link').id
   a=d.xaxis[j].copy();normal=np.array([0.,0.,1.]);lat=a.copy();lat[2]=0;lat/=np.linalg.norm(lat);roll=np.cross(lat,normal)
   radial=normal-a[2]*a;radial=-RADIUS*radial/np.linalg.norm(radial)
   cp=d.xanchor[j]+radial;J=np.zeros((3,m.nv));mujoco.mj_jac(m,d,J,None,cp,b)
   cv=J@v[k]
   angle=np.degrees(np.arctan2(np.dot(roll,np.cross(normal,forward)),np.dot(roll,forward)))
   row.append([cv@lat,cv@roll,np.linalg.norm(cv[:2]),cp[2],angle])
  out.append(row)
 return np.array(out)

def main():
 m=mujoco.MjModel.from_xml_path(MODEL);d=mujoco.MjData(m)
 old=np.load(OUT.parent/'resmimic_pick_place/go2w_carry_adapted.npz');q0=old['qpos'];t=np.arange(len(q0))/30
 # Euler is singular for an upright dog; use horizontal -Z direction instead.
 h=R.from_quat(q0[:,[4,5,6,3]]).apply(np.tile([0.,0.,-1.],(len(q0),1)))
 oldyaw=np.unwrap(np.arctan2(h[:,1],h[:,0]))
 upright=R.from_euler('y',-np.pi/2).as_quat()[[3,0,1,2]]
 rear=np.array([9,10,13,14]);hip=np.array([8,12]);wheel=np.array([3,7,11,15])
 limits=m.jnt_range[1:][rear];seed=np.array([2.,-1.5,2.,-1.5]);stances=[];stance_error=[]
 for i,orig in enumerate(q0):
  d.qpos[:]=orig;d.qpos[:2]=0;d.qpos[3:7]=upright;d.qpos[7+hip]=0;d.qpos[7+wheel]=0
  def fun(x):
   d.qpos[7+rear]=x;mujoco.mj_forward(m,d)
   l=d.body('RL_wheel_link').xpos;r=d.body('RR_wheel_link').xpos
   # equal axle X, wheel-ground height, unloaded static COM over axle.
   return np.r_[20*(l[2]-RADIUS),20*(r[2]-RADIUS),20*(l[0]-r[0]),10*(d.subtree_com[1,0]-(l[0]+r[0])/2)]
  sol=least_squares(fun,seed,bounds=(limits[:,0]+1e-5,limits[:,1]-1e-5),max_nfev=150,gtol=1e-11,xtol=1e-11,ftol=1e-11)
  seed=sol.x;fun(seed);stances.append(d.qpos.copy());stance_error.append(np.max(np.abs(fun(seed))))
 stances=np.array(stances)
 sx=[]
 for q in stances:d.qpos[:]=q;mujoco.mj_forward(m,d);sx.append(d.body('RL_wheel_link').xpos[0])
 sx=np.array(sx)
 # Move only after lift, stop before placing. Differential-drive axle twist is
 # integrated rather than copying lateral components from the human root path.
 ta,tb=2.65,5.6;knots=np.linspace(ta,tb,13)
 dense_t=np.arange(round(t[-1]*120)+1)/120
 sx_dense=CubicSpline(t,sx)(dense_t)
 yinit=np.interp(knots,t,oldyaw)
 p0=q0[0,:2]+sx[0]*np.array([np.cos(yinit[0]),np.sin(yinit[0])])
 def plan(x):
  yk=x[:13];vk=np.r_[0,x[13:],0]
  yc=CubicSpline(knots,yk,bc_type=((1,0.),(1,0.)))
  yd=yc(np.clip(dense_t,ta,tb));w=yc(np.clip(dense_t,ta,tb),1);w[(dense_t<ta)|(dense_t>tb)]=0
  speed=PchipInterpolator(knots,vk)(np.clip(dense_t,ta,tb))
  tangent=np.c_[np.cos(yd),np.sin(yd)]
  axle=p0+cumulative_trapezoid(speed[:,None]*tangent,dense_t,axis=0,initial=0)
  root=axle-sx_dense[:,None]*tangent
  return root,yd,speed,w,axle
 target=q0[:,:2]
 def objective(x):
  root,y,speed,w,axle=plan(x);rs=root[::4];ys=y[::4]
  # Preserve phase endpoints/trajectory; yaw is a soft preference, not a command
  # that can force sideways sliding.
  return np.r_[(rs-target).ravel()*3,(ys-oldyaw)*.025,np.diff(speed)*2,np.diff(w)*.08,
               (rs[-1]-target[-1])*80,(rs[0]-target[0])*80]
 x0=np.r_[yinit,np.full(11,.45)]
 sol=least_squares(objective,x0,bounds=(np.r_[yinit-.9,np.zeros(11)],np.r_[yinit+.9,np.ones(11)*1.5]),max_nfev=180,ftol=1e-8)
 root,y,speed,w,axle=plan(sol.x)
 q=PchipInterpolator(t,stances)(dense_t)
 q[:,:2]=root;q[:,3:7]=(R.from_euler('z',y)*R.from_euler('y',-np.pi/2)).as_quat()[:,[3,0,1,2]]
 # Interpolation can slightly lift wheel hubs: re-solve rear pitch at each dense
 # frame against the actual interpolated root height, keeping rear hips zero.
 # Analytic two-link geometry preserves equal X and exact wheel height.
 for k in range(len(q)):
  for rearbase in [8,12]:
   L1=.213;L2=.2264;dx=sx_dense[k];dz=RADIUS-(q[k,2]-.1934)
   # upright dog: x=L1*cos(thigh)+L2*cos(thigh+calf),
   # z=-L1*sin(thigh)-L2*sin(thigh+calf).
   c=np.clip((dx*dx+dz*dz-L1*L1-L2*L2)/(2*L1*L2),-1,1)
   knee=-np.arccos(c)
   thigh=np.arctan2(-dz,dx)-np.arctan2(L2*np.sin(knee),L1+L2*np.cos(knee))
   q[k,7+rearbase]=0;q[k,7+rearbase+1]=thigh;q[k,7+rearbase+2]=knee
 q[:,7+wheel]=0
 # wheel body pitch includes rear leg pitch. Cancel it in wheel joint phase.
 for side,idx,b in [(1,11,.142),(-1,15,-.142)]:
  phase=cumulative_trapezoid((speed-w*b)/RADIUS,dense_t,initial=0)
  base=8 if side==1 else 12
  legpitch=q[:,7+base+1]+q[:,7+base+2]
  q[:,7+idx]=phase-(legpitch-legpitch[0])
 qv=velocities(m,q,1/120);qa=np.gradient(qv,1/120,axis=0)
 metrics=audit(m,q,120);oldmetrics=audit(m,q0,30)
 carry=(dense_t>=100/30)&(dense_t<=140/30)
 def stats(a):return {'rms':float(np.sqrt(np.mean(a*a))),'max_abs':float(np.max(np.abs(a)))}
 ids=np.array([i for i in range(16) if i%4!=3])
 assert np.isfinite(q).all()
 assert np.all(q[:,7+ids]>=m.jnt_range[1:][ids,0]-1e-6)
 print('limit excess',np.max(q[:,7+ids]-m.jnt_range[1:][ids,1]),flush=True)
 assert np.all(q[:,7+ids]<=m.jnt_range[1:][ids,1]+1e-6)
 assert np.max(np.abs(metrics[:,:,0]))<.002
 assert np.max(metrics[:,:,2])<.003
 assert np.max(np.abs(metrics[:,:,3]))<1e-6
 report=dict(method='parallel rear wheel stance + integrated differential-drive trajectory + leg-pitch compensated wheel phase',fps=120,frames=len(q),duration_s=float(t[-1]),transport_interval_s=[ta,tb],stance_equation_max_residual=float(np.max(stance_error)),whole_clip_lateral_slip_m_s=stats(metrics[:,:,0]),whole_clip_planar_slip_m_s=stats(metrics[:,:,2]),carry_lateral_slip_m_s=stats(metrics[carry,:,0]),old_carry_lateral_slip_m_s=stats(oldmetrics[100:141,:,0]),toe_angle_max_deg=float(np.max(np.abs(metrics[:,:,4]))),ground_gap_max_m=float(np.max(np.abs(metrics[:,:,3]))),root_path_change_m=stats(np.linalg.norm(root[::4]-target,axis=1)),root_end_error_m=float(np.linalg.norm(root[-1]-target[-1])),root_heading_change_deg=stats(np.degrees(y[::4]-oldyaw)),rear_joint_max_speed_rad_s=float(np.max(np.abs(qv[:,[m.jnt_dofadr[i+1] for i in ids if i>=8]]))),kinematic_screen='PASS ideal circular wheel / level-ground assumptions',dynamics='not yet evaluated',object_load='not modeled',front_leg_joint_reference='preserved by spline interpolation from prior pick-place prototype')
 np.savez(OUT/'reference.npz',fps=120,qpos=q,qvel=qv,qacc=qa,time=dense_t,root_pos=q[:,:3],root_rot=q[:,[4,5,6,3]],dof_pos=q[:,7:],wheel_metrics=metrics,axle_xy=axle,forward_speed=speed,yaw_rate=w,joint_names=np.array([m.joint(i).name for i in range(1,m.njnt)]))
 with open(OUT/'reference.pkl','wb') as f:pickle.dump(dict(fps=120,root_pos=q[:,:3],root_rot=q[:,[4,5,6,3]],dof_pos=q[:,7:],local_body_pos=None,link_body_list=None),f)
 (OUT/'kinematic_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
