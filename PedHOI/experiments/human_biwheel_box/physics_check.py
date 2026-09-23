"""Independent carry-only free-payload stress test. Never used to animate the reference."""
from pathlib import Path
import numpy as np,mujoco,json
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R
from scipy.linalg import solve_continuous_are
P=Path(__file__).resolve().parent;z=np.load(P/'reference.npz');tt=z['time'];qr=PchipInterpolator(tt,z['qpos']);vr=PchipInterpolator(tt,z['qvel']);br=PchipInterpolator(tt,z['object_pos']);bvr=br.derivative()
m=mujoco.MjModel.from_xml_path(str(P/'scene.xml'));m.opt.timestep=.002;m.opt.iterations=80;d=mujoco.MjData(m)
robot=mujoco.MjModel.from_xml_path('/home/robotennis2025/SJTU-BipedGo2w/GMR/assets/unitree_go2w/go2w.xml');rd=mujoco.MjData(robot)
S=np.zeros((22,16))
for i in range(16):S[robot.jnt_dofadr[robot.actuator_trnid[i,0]],i]=robot.actuator_gear[i,0]
feed=[]
for k in range(0,len(tt),8):
 rd.qpos[:]=z['qpos'][k];rd.qvel[:]=z['qvel'][k];mujoco.mj_forward(robot,rd);M=np.zeros((22,22));mujoco.mj_fullM(robot,M,rd.qM);rhs=M@z['qacc'][k]+rd.qfrc_bias-rd.qfrc_passive
 Jrear=[]
 for leg in ['RL','RR']:
  b=robot.body(leg+'_wheel_link').id;J=np.zeros((3,22));mujoco.mj_jac(robot,rd,J,None,rd.xpos[b]-[0,0,.086],b);Jrear.append(J)
 for leg in ['FL','FR']:
  b=robot.body(leg+'_wheel_link').id;J=np.zeros((3,22));mujoco.mj_jac(robot,rd,J,None,rd.xpos[b]+[0,0,.086],b);rhs+=J.T@np.array([0,0,.5*9.81/2])
 mat=np.c_[S,Jrear[0].T,Jrear[1].T];feed.append(np.linalg.lstsq(mat,rhs,rcond=None)[0][:16])
ff=PchipInterpolator(tt[::8],feed)
M,mb,l=4.,15.,.4;g=9.81
A=np.array([[0,1,0,0],[0,0,-mb*g/M,0],[0,0,0,1],[0,0,(M+mb)*g/(M*l),0]]);B=np.array([[0],[1/M],[0],[-1/(M*l)]])
K=np.linalg.solve(np.array([[.1]]),B.T@solve_continuous_are(A,B,np.diag([5,8,180,8]),np.array([[.1]])));K*=np.array([[8,4,4,2]])
start,end=6.8,17.46;d.qpos[:23]=qr(start);d.qvel[:22]=vr(start);d.qpos[23:26]=br(start);d.qpos[26:]=z['object_quat'][round(start*120)];d.qvel[22:25]=bvr(start);d.qvel[25:28]=[0,0,z['yaw_rate'][round(start*120)]];mujoco.mj_forward(m,d)
records=[];controls=[];contactrows=[];failure=None;boxgeom=m.geom('payload_box').id
for k in range(round((end-start)/.002)):
 t=start+k*.002;qref=qr(t);vref=vr(t);rr=R.from_quat(qref[[4,5,6,3]]);ra=R.from_quat(d.qpos[[4,5,6,3]]);fw=rr.apply([0,0,-1]);fw[2]=0;fw/=np.linalg.norm(fw);lat=np.cross([0,0,1],fw)
 tilt=np.arctan2(ra.apply([1,0,0])@fw,ra.apply([1,0,0])[2]);omega=ra.apply(d.qvel[3:6])@lat
 state=np.array([(d.qpos[:3]-qref[:3])@fw,(d.qvel[:3]-vref[:3])@fw,tilt,omega]);common=float((-K@state).item())*.086/2
 ye=(ra*rr.inv()).as_rotvec()[2];yv=ra.apply(d.qvel[3:6])[2]-rr.apply(vref[3:6])[2];ds=np.sign(vref[:3]@fw) if abs(vref[:3]@fw)>.02 else 0;ye+=np.clip(1.5*((d.qpos[:3]-qref[:3])@lat)*ds,-.4,.4);yt=-40*ye-8*yv
 baseff=ff(t)
 for i in range(m.nu):
  j=m.actuator_trnid[i,0];qi=m.jnt_qposadr[j];vi=m.jnt_dofadr[j];name=m.joint(j).name
  if name.startswith('RL_wheel'):tau=common-yt*.086/(2*.142)
  elif name.startswith('RR_wheel'):tau=common+yt*.086/(2*.142)
  elif 'wheel' in name:tau=baseff[i]-1.5*(d.qvel[vi]-vref[vi])
  else:tau=baseff[i]+100*(qref[qi]-d.qpos[qi])+5*(vref[vi]-d.qvel[vi])
  d.ctrl[i]=np.clip(tau,*m.actuator_ctrlrange[i])
 mujoco.mj_step(m,d)
 if k%5==0:
  contact=[False,False]
  for c in d.contact:
   if boxgeom in [c.geom1,c.geom2]:
    other=c.geom2 if c.geom1==boxgeom else c.geom1
    for i,leg in enumerate(['FL','FR']):
     if m.geom_bodyid[other]==m.body(leg+'_wheel_link').id and c.dist<.001:contact[i]=True
  records.append(np.r_[t,d.qpos.copy(),d.qvel.copy()]);controls.append(d.ctrl.copy());contactrows.append(contact)
 # Detect loss relative to robot, not just error relative to global reference.
 target=d.qpos[:3]+ra.apply(rr.inv().apply(br(t)-qref[:3]));be=np.linalg.norm(d.qpos[23:26]-target)
 if d.qpos[2]<.25 or np.linalg.norm(d.qpos[:3]-qref[:3])>.5 or be>.18:
  failure=dict(time_s=t,box_relative_position_error_m=float(be),root_z_m=float(d.qpos[2]));break
np.savez(P/'free_box_rollout.npz',records=records,ctrl=controls,front_box_contacts=contactrows)
report=dict(test='carry-only stress test, initialized at start of carry; not pick/place validation',payload_mass_kg=.5,completed=failure is None,elapsed_s=(k+1)*.002,planned_s=end-start,failure=failure,front_box_contact_fraction=np.mean(contactrows,axis=0).tolist(),root_wrench=False,pose_resets=False,weld_or_mocap=False,controller='existing balance controller with new arm feedforward and approximate payload gravity; no object feedback')
(P/'physics_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
