"""Raw human -> phase-dependent foreleg IK -> rolling-constrained bi-wheel ref.
No G1 reference is read. World objects are source evidence, not simulated loads.
"""
from pathlib import Path
import sys,json,pickle,os
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline,PchipInterpolator
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'resmimic_rolling_v2'))
from build_reference import velocities,audit,MODEL,RADIUS
P=Path(os.environ.get("PEDHOI_STAGE_DIR",Path(__file__).resolve().parent))
TIME_SCALE=4.;FPS=120
m=mujoco.MjModel.from_xml_path(MODEL);d=mujoco.MjData(m)
h=np.load(P/'human_source.npz');meta=json.loads((P/'source.json').read_text());j=h['joints_w'];N=len(j);ts=np.arange(N)/30;td=np.arange(round(ts[-1]*TIME_SCALE*FPS)+1)/FPS;source_t=td/TIME_SCALE
phase=meta['phases_zero_based'];cs,ce=phase['carry_start'],phase['carry_end']
side=j[:,1]-j[:,2];side[:,2]=0;side/=np.linalg.norm(side,axis=1)[:,None];forward=np.cross(side,[0,0,1]);yaw=np.unwrap(np.arctan2(forward[:,1],forward[:,0]))
H=R.from_euler('z',yaw).as_matrix();rel=np.einsum('nji,nkj->nki',H,j-j[:,:1])
# Morphology-specific bilateral contact proxy; preserve reach/height timing.
hands=rel[:,[20,21]].copy();hands[:,:,0]=.50*hands[:,:,0]+.06;hands[:,:,1]=.6*hands[:,:,1];hands[:,:,2]=.55*hands[:,:,2]+.055
hands[:,0,1]=np.clip(hands[:,0,1],.12,.24);hands[:,1,1]=np.clip(hands[:,1,1],-.24,-.12)
def gate(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
blend=gate((np.arange(N)-(cs-12))/12)*(1-gate((np.arange(N)-ce)/12))
hold=np.median(hands[cs+8:ce-8],axis=0);hands=hands*(1-blend[:,None,None])+hold[None]*blend[:,None,None]
height=np.clip(.55+.25*(j[:,0,2]-np.median(j[cs:ce,0,2])),.46,.56)
height=height*(1-blend)+.55*blend
qsource=np.zeros((N,m.nq));qsource[:,2]=height;qsource[:,3:7]=R.from_euler('y',-np.pi/2).as_quat()[[3,0,1,2]];qsource[:,7:]=np.tile([0,.8,-1.5,0],4)
fi=np.array([0,1,2,4,5,6]);ri=np.array([9,10,13,14]);limits=m.jnt_range[1:];seed=qsource[0,7+fi];rseed=np.array([2.,-1.5,2.,-1.5]);sx=[];err=[]
for f in range(N):
 d.qpos[:]=qsource[f]
 def front(x):
  d.qpos[7+fi]=x;mujoco.mj_forward(m,d)
  return np.r_[(d.xpos[[m.body('FL_wheel_link').id,m.body('FR_wheel_link').id]]-(hands[f]+[0,0,height[f]])).ravel(),.012*(x-seed)]
 lo=limits[fi,0]+1e-5;hi=limits[fi,1]-1e-5
 if f:lo=np.maximum(lo,seed-.12);hi=np.minimum(hi,seed+.12)
 res=least_squares(front,np.clip(seed,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=100);seed=res.x;front(seed);err.append(np.linalg.norm((d.xpos[[m.body('FL_wheel_link').id,m.body('FR_wheel_link').id]]-(hands[f]+[0,0,height[f]])),axis=1))
 def stance(x):
  d.qpos[7+ri]=x;d.qpos[[15,19]]=0;mujoco.mj_forward(m,d);l=d.body('RL_wheel_link').xpos;r=d.body('RR_wheel_link').xpos
  return [l[2]-RADIUS,r[2]-RADIUS,l[0]-r[0],d.subtree_com[1,0]-(l[0]+r[0])/2]
 res=least_squares(stance,rseed,bounds=(limits[ri,0]+1e-5,limits[ri,1]-1e-5),gtol=1e-12,xtol=1e-12,ftol=1e-12);rseed=res.x;stance(rseed);sx.append(d.body('RL_wheel_link').xpos[0]);qsource[f]=d.qpos.copy()
sx=np.array(sx);sx_dense=PchipInterpolator(ts,sx)(source_t)
# Differential-drive planning. Signed wheel travel permits backwards carrying.
ta=max(0,(cs-8)/30);tb=min(ts[-1],(ce+10)/30);knots=np.linspace(ta,tb,17);yk0=np.interp(knots,ts,yaw)
target=j[:,0,:2]-j[0,0,:2]
yaw_start=yaw[0];yaw_end=yaw[-1]
p0=sx[0]*np.array([np.cos(yaw_start),np.sin(yaw_start)])
def plan(x):
 yk=x[:17];vk=np.r_[0,x[17:],0]
 yy=CubicSpline(knots,yk,bc_type=((1,0),(1,0)))(np.clip(source_t,ta,tb))
 # Approach/retraction may rotate in place to honor interaction heading.
 pre=source_t<ta;post=source_t>tb
 yy[pre]=yaw_start+(yk[0]-yaw_start)*gate(source_t[pre]/ta)
 yy[post]=yk[-1]+(yaw_end-yk[-1])*gate((source_t[post]-tb)/(ts[-1]-tb))
 speed=PchipInterpolator(knots,vk)(np.clip(source_t,ta,tb))/TIME_SCALE
 tangent=np.c_[np.cos(yy),np.sin(yy)]
 axle=p0+cumulative_trapezoid(speed[:,None]*tangent,td,axis=0,initial=0)
 root=axle-sx_dense[:,None]*tangent
 return root,yy,speed,axle
# init signed forward projection of original velocity
vel=np.gradient(target,1/30,axis=0);speed0=np.sum(vel*np.c_[np.cos(yaw),np.sin(yaw)],axis=1)
x0=np.r_[yk0,np.clip(np.interp(knots[1:-1],ts,speed0),-2,2)]
def obj(x):
 root,yy,speed,axle=plan(x);rs=root[::16];ys=yy[::16]
 return np.r_[(rs-target).ravel()*2,(ys-yaw)*.15,np.diff(x[:17],2)*.1,np.diff(x[17:])*.05,(rs[-1]-target[-1])*60]
fit=least_squares(obj,x0,bounds=(np.r_[yk0-1.3,np.full(15,-2.)],np.r_[yk0+1.3,np.full(15,2.)]),max_nfev=150)
root,yd,speed,axle=plan(fit.x);omega=np.gradient(yd,1/FPS);tangent=np.c_[np.cos(yd),np.sin(yd)]
qbase=PchipInterpolator(ts,qsource)(source_t);ids=np.array([i for i in range(16) if i%4!=3])
# Contact balance correction: shift axle relative to root, using free-base
# resultant wrench. Low pass corrections because qdd is numerically sensitive.
def construct(offset):
 q=qbase.copy();xx=sx_dense+offset;q[:,:2]=axle-xx[:,None]*tangent
 q[:,3:7]=(R.from_euler('z',yd)*R.from_euler('y',-np.pi/2)).as_quat()[:,[3,0,1,2]]
 for base in [8,12]:
  dz=RADIUS-(q[:,2]-.1934);cc=(xx*xx+dz*dz-.213**2-.2264**2)/(2*.213*.2264)
  knee=-np.arccos(np.clip(cc,-1,1));thigh=np.arctan2(-dz,xx)-np.arctan2(.2264*np.sin(knee),.213+.2264*np.cos(knee))
  q[:,7+base]=0;q[:,7+base+1]=thigh;q[:,7+base+2]=knee
 for b,widx,base in [(.142,11,8),(-.142,15,12)]:
  wheelphase=cumulative_trapezoid((speed-omega*b)/RADIUS,td,initial=0)
  pitch=q[:,7+base+1]+q[:,7+base+2];q[:,7+widx]=wheelphase-(pitch-pitch[0])
 return q
balance_history=[];offset=np.zeros(len(td));mass=m.body_mass.sum()
for it in range(5):
 q=construct(offset);v=velocities(m,q,1/FPS);acc=np.gradient(v,1/FPS,axis=0);correction=[];residual=[]
 for k in range(0,len(q),4):
  d.qpos[:]=q[k];d.qvel[:]=v[k];mujoco.mj_forward(m,d);M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,M,d.qM);rhs=M@acc[k]+d.qfrc_bias-d.qfrc_passive
  force=rhs[:3];torque=R.from_quat(q[k,[4,5,6,3]]).apply(rhs[3:6]);center=(d.body('RL_wheel_link').xpos+d.body('RR_wheel_link').xpos)/2;center[2]-=RADIUS
  lat=np.array([-np.sin(yd[k]),np.cos(yd[k]),0]);e=(torque-np.cross(center-q[k,:3],force))@lat
  correction.append(-e/max(force[2],50));residual.append(e)
 balance_history.append(float(np.sqrt(np.mean(np.square(residual)))))
 if it<4:
  delta=np.interp(td,td[::4],correction);delta=gaussian_filter1d(delta,20)
  offset=np.clip(offset+.5*delta,-.05,.05)
metrics=audit(m,q,FPS);carry=(source_t>=cs/30)&(source_t<=ce/30)
# Static geometry check and framewise statistics. Dynamic acceptance is separate.
assert np.isfinite(q).all();assert np.all(q[:,7+ids]>=limits[ids,0]-1e-6);assert np.all(q[:,7+ids]<=limits[ids,1]+1e-6)
def stats(x):return dict(rms=float(np.sqrt(np.mean(x*x))),max_abs=float(np.max(np.abs(x))))
np.savez(P/'reference.npz',fps=FPS,qpos=q,qvel=v,qacc=acc,time=td,source_time=source_t,root_pos=q[:,:3],root_rot=q[:,[4,5,6,3]],dof_pos=q[:,7:],wheel_metrics=metrics,axle_xy=axle,forward_speed=speed,yaw_rate=omega,front_targets_local=hands,source_front_ik_error=np.array(err),balance_offset=offset,phase_blend=np.interp(source_t,ts,blend),joint_names=np.array([m.joint(i).name for i in range(1,m.njnt)]))
with open(P/'reference.pkl','wb') as f:pickle.dump(dict(fps=FPS,root_pos=q[:,:3],root_rot=q[:,[4,5,6,3]],dof_pos=q[:,7:],local_body_pos=None,link_body_list=None),f)
report=dict(source_sequence=meta['sequence'],human_input='raw OMOMO -> SMPL-X joints, no robot intermediate',frames=len(q),fps=FPS,duration_s=float(td[-1]),time_scale=TIME_SCALE,phase_source_frames=phase,front_target_fit_m=stats(np.array(err)),no_slip_whole_clip_m_s=stats(metrics[:,:,2]),no_slip_carry_m_s=stats(metrics[carry,:,2]),toe_angle_max_deg=float(np.max(np.abs(metrics[:,:,4]))),ground_gap_max_m=float(np.max(np.abs(metrics[:,:,3]))),path_change_m=stats(np.linalg.norm(q[::16,:2]-target,axis=1)),heading_change_deg=stats(np.degrees(yd[::16]-yaw)),end_position_error_m=float(np.linalg.norm(q[-1,:2]-target[-1])),balance_iteration_axle_moment_rms_Nm=balance_history,balance_offset_max_m=float(np.max(np.abs(offset))),kinematic_pass=bool(np.max(metrics[:,:,2])<.003 and np.max(np.abs(metrics[:,:,3]))<1e-5),scope='robot-only reference, hand targets morphology-scaled; object trajectory not a physical payload; no trained policy')
(P/'retarget_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
