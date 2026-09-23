"""Human-anchored ground-to-ground lateral clamp with real wheel surfaces."""
from pathlib import Path
import sys,json,os,xml.etree.ElementTree as ET
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'resmimic_rolling_v2'))
from build_reference import MODEL,velocities,audit
P=Path(os.environ.get('PEDHOI_STAGE_DIR',Path(__file__).resolve().parent));OLD=Path(os.environ.get('PEDHOI_SEED_DIR',Path(__file__).resolve().parent.parent/'human_biwheel'));h=np.load(OLD/'human_source.npz');old=dict(np.load(OLD/'reference.npz'));j=h['joints_w'];src=np.arange(len(j))/30;tt=old['time'];st=old['source_time'];size=np.array([.12,.16,.13]);mass=.5
m=mujoco.MjModel.from_xml_path(MODEL);d=mujoco.MjData(m)
side=j[:,1]-j[:,2];side[:,2]=0;side/=np.linalg.norm(side,axis=1)[:,None];hf=np.cross(side,[0,0,1]);spine=j[:,12]-j[:,0];humanlean=np.arctan2((spine*hf).sum(1),spine[:,2]);lean=np.clip(.65*humanlean,0,np.radians(52));height=np.clip(.55+.35*(j[:,0,2]-.918),.40,.56)
# Real source object lift curve, not a constant carry plateau.
sourceground=np.median(h['object_pos'][:25,2]);lift=np.maximum(h['object_pos'][:,2]-sourceground,0)
# Scanned source rest height has millimetric drift: explicitly pin settled endpoints.
lift[:32]=0;lift[156:]=0
objheight=size[2]+.43*lift
rel=h['object_pos']-j[:,0];reach=np.clip(.5*(rel*hf).sum(1),.23,.34)
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
closed=smooth((np.arange(len(j))-12)/18)*(1-smooth((np.arange(len(j))-156)/23));gap=.045*(1-closed)
# Open hands follow human wrist withdrawal height; closed hands stay on box sides.
wrists=j[:,[20,21],2].mean(1);withdraw=.6*np.maximum(wrists-(h['object_pos'][:,2]+.20),0)*(1-closed)
fi=np.array([0,1,2,4,5,6]);ri=np.array([9,10,13,14]);ids=np.r_[fi,ri];lim=m.jnt_range[1:];seed=np.r_[old['qpos'][0,7+fi],[2,-1.5,2,-1.5],-.05];lo=np.r_[lim[ids,0]+1e-6,-.24];hi=np.r_[lim[ids,1]-1e-6,.08]
gids=[];verts=[]
for leg in ['FL','FR']:
 g=np.where(m.geom_bodyid==m.body(leg+'_wheel_link').id)[0][0];gids.append(g);mid=m.geom_dataid[g];verts.append(m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]].copy())
def surfaces():
 out=[]
 for i,(g,v) in enumerate(zip(gids,verts)):
  w=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];center=(w.max(0)+w.min(0))/2;center[1]=w[:,1].min() if i==0 else w[:,1].max();out.append(center)
 return np.array(out)
qsource=[];errors=[];stanceerrors=[]
for k in range(len(j)):
 q=old['qpos'][k*16].copy();q[:3]=[0,0,height[k]];q[3:7]=R.from_euler('y',-np.pi/2+lean[k]).as_quat()[[3,0,1,2]];q[[15,19]]=0;d.qpos[:]=q
 target=np.array([[reach[k]-.09,size[1]+gap[k],objheight[k]+size[2]+.04+withdraw[k]],[reach[k]-.09,-size[1]-gap[k],objheight[k]+size[2]+.04+withdraw[k]]])
 def fun(x):
  d.qpos[7+ids]=x[:10];d.qpos[0]=x[10];mujoco.mj_forward(m,d);rear=d.xpos[[m.body('RL_wheel_link').id,m.body('RR_wheel_link').id]];com=(d.subtree_com[1,0]*m.body_mass.sum()+closed[k]*mass*reach[k])/(m.body_mass.sum()+closed[k]*mass)
  return np.r_[(surfaces()-target).ravel()*2,(rear[:,[0,2]]-[[0,.086],[0,.086]]).ravel()*3,com*1.5,.00015*(x-seed)]
 fit=least_squares(fun,np.clip(seed,lo+1e-7,hi-1e-7),bounds=(lo,hi),max_nfev=150,ftol=1e-11,xtol=1e-11,gtol=1e-11);seed=fit.x;rr=fun(seed);qsource.append(d.qpos.copy());errors.append(np.linalg.norm(surfaces()-target,axis=1));stanceerrors.append(rr[6:11]);
qsource=np.array(qsource);q=PchipInterpolator(src,qsource)(st);pitch=PchipInterpolator(src,-np.pi/2+lean)(st)
fw=R.from_quat(old['qpos'][:,[4,5,6,3]]).apply(np.tile([0,0,-1.],(len(tt),1)));yaw=np.unwrap(np.arctan2(fw[:,1],fw[:,0]));H=R.from_euler('z',yaw)
# Existing differential-drive axle path is preserved, with no added reverse detour.
axle=old['axle_xy'];q[:,:2]=axle+q[:,0,None]*fw[:,:2];q[:,3:7]=(H*R.from_euler('y',pitch)).as_quat()[:,[3,0,1,2]]
# Enforce exact rear contact after interpolation using planar analytic IK.
for base,w in [(8,11),(12,15)]:
 dx=-PchipInterpolator(src,qsource[:,0])(st);dz=.086-q[:,2];chainx=np.cos(pitch)*dx-np.sin(pitch)*dz+.1934;chainz=np.sin(pitch)*dx+np.cos(pitch)*dz
 c=(chainx*chainx+chainz*chainz-.213**2-.2264**2)/(2*.213*.2264);knee=-np.arccos(np.clip(c,-1,1));thigh=np.arctan2(-chainx,-chainz)-np.arctan2(.2264*np.sin(knee),.213+.2264*np.cos(knee));q[:,7+base+1]=thigh;q[:,7+base+2]=knee
 # Preserve integrated wheel travel and cancel changing torso/leg pitch.
 oldpitch=old['qpos'][:,7+base+1]+old['qpos'][:,7+base+2]-np.pi/2;newpitch=pitch+thigh+knee
 q[:,7+w]=old['qpos'][:,7+w]-(newpitch-newpitch[0])+(oldpitch-oldpitch[0])
box=np.c_[axle,np.zeros(len(tt))]+H.apply(np.c_[PchipInterpolator(src,reach)(st),np.zeros(len(tt)),PchipInterpolator(src,objheight)(st)])
# Ground endpoints are world-fixed; solve approach/retraction IK to those boxes below.
pick=31*16;place=156*16;box[:pick]=box[pick];box[place:]=box[place];byaw=yaw.copy();byaw[:pick]=yaw[pick];byaw[place:]=yaw[place]
# Correct front IK at full output rate, accounting for world-fixed endpoints.
errdense=[];targetsdense=[];seed=q[0,7+fi].copy();gapdense=PchipInterpolator(src,gap)(st);withdrawdense=PchipInterpolator(src,withdraw)(st)
for k in range(len(q)):
 d.qpos[:]=q[k];bh=R.from_euler('z',byaw[k]);tgt=box[k]+bh.apply(np.array([[-.09,size[1]+gapdense[k],size[2]+.04+withdrawdense[k]],[-.09,-size[1]-gapdense[k],size[2]+.04+withdrawdense[k]]]))
 def surfworld():
  out=[]
  for i,(g,v) in enumerate(zip(gids,verts)):
   w=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];local=(w-box[k])@bh.as_matrix();center=(local.max(0)+local.min(0))/2;center[1]=local[:,1].min() if i==0 else local[:,1].max();out.append(box[k]+bh.apply(center))
  return np.array(out)
 def fun(x):
  d.qpos[7+fi]=x;mujoco.mj_forward(m,d);return np.r_[(surfworld()-tgt).ravel(),.0001*(x-seed)]
 fit=least_squares(fun,np.clip(seed,lim[fi,0]+1e-6,lim[fi,1]-1e-6),bounds=(lim[fi,0]+1e-6,lim[fi,1]-1e-6),max_nfev=60,gtol=1e-10,ftol=1e-10,xtol=1e-10);seed=fit.x;fun(seed);q[k]=d.qpos;errdense.append(np.linalg.norm(surfworld()-tgt,axis=1));targetsdense.append(tgt)
v=velocities(m,q,1/120);a=np.gradient(v,1/120,axis=0);metrics=audit(m,q,120)
z=dict(fps=120,qpos=q,qvel=v,qacc=a,time=tt,source_time=st,root_pos=q[:,:3],root_rot=q[:,[4,5,6,3]],dof_pos=q[:,7:],object_pos=box,object_quat=R.from_euler('z',byaw).as_quat()[:,[3,0,1,2]],object_half_size=size,object_mass=mass,contact_targets=np.array(targetsdense),jaw_gap=gapdense,grip_blend=PchipInterpolator(src,closed)(st),wheel_metrics=metrics,axle_xy=axle,forward_speed=old['forward_speed'],yaw_rate=old['yaw_rate'],human_lean=humanlean,human_object_height=h['object_pos'][:,2],training_ready=False,payload_manipulation_validated=False,joint_names=old['joint_names'])
np.savez(P/'reference.npz',**z)
tree=ET.parse(MODEL);root=tree.getroot();root.find('compiler').set('meshdir',str(Path(MODEL).parent/'assets'));world=root.find('worldbody');b=ET.SubElement(world,'body',name='payload',pos=' '.join(map(str,box[0])));ET.SubElement(b,'freejoint',name='payload_free');ET.SubElement(b,'geom',name='payload_box',type='box',size=' '.join(map(str,size)),mass=str(mass),rgba='.65 .36 .13 1',friction='.8 .02 .01',condim='6');tree.write(P/'scene.xml')
report=dict(scope='human-anchored ground pick/clamp/carry/place kinematic reference',box_dimensions_m=(2*size).tolist(),mass_assumption_kg=mass,source_sequence='sub16_largebox_010',source_lift_start_frame=34,source_last_elevated_frame=153,robot_ground_before_source_frame=31,robot_ground_after_source_frame=156,object_lift_vertical_scale=.43,torso_lean_scale=.65,extra_reverse_detour=False,platforms=False,front_surface_fit_max_m=float(np.max(errdense)),source_stance_error_max=float(np.max(np.abs(stanceerrors))),rear_no_slip_max_m_s=float(metrics[:,:,2].max()),rear_ground_gap_max_m=float(np.max(np.abs(metrics[:,:,3]))),object_ground_start_end_m=(box[[0,-1],2]-size[2]).tolist(),geometry_validated=False,dynamics_validated=False)
(P/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
