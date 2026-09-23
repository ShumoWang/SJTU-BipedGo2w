"""Joint object SE(3)/front-leg correction with interaction-mesh residuals.
Rear rolling solution is immutable in this stage. All acceptance is independent.
"""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R,Slerp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from .object_prior import corners
from .geometry import pair_distance

def run(cfg,out,seedpath):
 out=Path(out);z=dict(np.load(seedpath));prior=np.load(out/'interaction.npz');tt=z['time'];st=z['source_time'];half=np.array(cfg['box_dimensions_m'])/2;N=len(tt);nf=len(prior['source_clearance']);ts=np.arange(nf)/cfg['source_fps'];source_idx=np.rint(ts*cfg['time_scale']*cfg['output_fps']).astype(int)
 tree=ET.parse(cfg['robot_xml']);root=tree.getroot();root.find('compiler').set('meshdir',str(Path(cfg['robot_xml']).parent/'assets'));body=ET.SubElement(root.find('worldbody'),'body',name='payload');ET.SubElement(body,'freejoint',name='payload_free');ET.SubElement(body,'geom',name='payload_box',type='box',size=' '.join(map(str,half)),mass=str(cfg['box_mass_kg']),rgba='.65 .36 .13 1',friction='.8 .02 .01',condim='6');tree.write(out/'scene.xml')
 m=mujoco.MjModel.from_xml_path(str(out/'scene.xml'));d=mujoco.MjData(m);obj=m.geom('payload_box').id;fi=np.array([0,1,2,4,5,6]);limits=m.jnt_range[1:17];q0=z['qpos'];baseR=R.from_quat(q0[:,[4,5,6,3]]);fw=baseR.apply(np.tile([0,0,-1.],(N,1)));yaw=np.unwrap(np.arctan2(fw[:,1],fw[:,0]));HR=R.from_euler('z',yaw)
 relative=R.from_quat(prior['source_object_relative_quat'][:,[1,2,3,0]]);fullR=HR[source_idx]*relative
 # Ground-rest pose has zero roll/pitch and is fixed in the world until lift/release.
 def upright(rot):mat=rot.as_matrix();return R.from_euler('z',np.arctan2(mat[1,0],mat[0,0]))
 firstR=upright(fullR[0]);lastR=upright(fullR[156]);pnom=z['object_pos'][source_idx].copy();pnom[:32,:2]=pnom[31,:2];pnom[156:,:2]=pnom[156,:2]
 gids=[];verts=[];wheelbodies=[]
 for leg in ['FL','FR']:
  b=m.body(leg+'_wheel_link').id;wheelbodies.append(b);g=np.where(m.geom_bodyid==b)[0][0];gids.append(g);mid=m.geom_dataid[g];verts.append(m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]].copy())
 elbowids=[m.body('FL_calf').id,m.body('FR_calf').id];neckids=[m.body('FL_hip').id,m.body('FR_hip').id]
 target_corners=corners(half);limit=np.radians(cfg['object_rotation_limit_deg']);positionlimit=cfg['object_position_correction_limit_m'];qout=[];pout=[];rout=[];stats=[];prev=q0[0,7+fi].copy();prev_r=np.zeros(3);prev_delta=np.zeros(2)
 def evaluate(q,rot,pos,x):
  d.qpos[:23]=q;d.qpos[7+fi]=x;d.qpos[23:26]=pos;d.qpos[26:]=rot.as_quat()[[3,0,1,2]];mujoco.mj_forward(m,d)
  surface=[]
  for i,(g,v) in enumerate(zip(gids,verts)):
   w=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];local=(w-pos)@rot.as_matrix();center=(local.min(0)+local.max(0))/2;center[1]=local[:,1].min() if i==0 else local[:,1].max();surface.append(center)
  distances=[]
  for g in gids:
   distances.append(pair_distance(m,d,g,obj))
  boxpen=0.;selfpen=0.;floorpen=0.
  for c in d.contact:
   a,b=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
   if obj in [c.geom1,c.geom2]:
    other=c.geom2 if c.geom1==obj else c.geom1
    if m.geom_bodyid[other] not in wheelbodies:boxpen=max(boxpen,-c.dist)
   elif a and b:selfpen=max(selfpen,-c.dist)
   elif a in wheelbodies or b in wheelbodies:floorpen=max(floorpen,-c.dist)
  nodes=np.vstack([d.qpos[:3],d.xpos[neckids].mean(0),d.xpos[elbowids],pos+rot.apply(surface)])
  localnodes=rot.inv().apply(nodes-pos);vertices=np.vstack([localnodes,target_corners])
  return np.array(surface),np.array(distances),np.array([boxpen,selfpen,floorpen]),vertices
 for k,ix in enumerate(source_idx):
  q=q0[ix].copy();active=32<=k<156;gap=float(z['jaw_gap'][ix]);withdraw=max(float(z['contact_targets'][ix,0,2]-z['object_pos'][ix,2]-half[2]-.04),0)
  target=np.array([[-.09,half[1]+gap,half[2]+.04+withdraw],[-.09,-half[1]-gap,half[2]+.04+withdraw]])
  wanted=relative[k].as_rotvec();wanted*=min(1,limit/max(np.linalg.norm(wanted),1e-9));initial=np.r_[prev,wanted if k==32 else prev_r,prev_delta] if active else prev
  low=np.r_[limits[fi,0]+1e-6,np.full(3,-limit),np.full(2,-positionlimit)] if active else limits[fi,0]+1e-6;high=np.r_[limits[fi,1]-1e-6,np.full(3,limit),np.full(2,positionlimit)] if active else limits[fi,1]-1e-6
  def unpack(x):
   rot=HR[ix]*R.from_rotvec(x[6:9]) if active else (firstR if k<32 else lastR)
   p=pnom[k].copy()
   if active:p[:2]+=HR[ix].apply([*x[9:11],0])[:2]
   p[2]=np.abs(rot.as_matrix()[2])@half+cfg['clearance_scale']*prior['source_clearance'][k]
   return rot,p
  def fun(x):
   rot,p=unpack(x);surface,dist,pen,vertices=evaluate(q,rot,p,x[:6]);contact_on=float(z['grip_blend'][ix]);lap=prior['laplacian'][k]@vertices-prior['target_laplacian'][k]
   residual=[(surface-target).ravel(),dist*5*contact_on,100*np.minimum(dist+1e-5,0),100*np.maximum(pen-1e-5,0),cfg['laplacian_weight']*lap.ravel(),.001*(x[:6]-prev)]
   if active:residual.extend([.04*(rot.inv()*fullR[k]).as_rotvec(),.8*x[9:11],.008*(x[6:9]-prev_r),[max(np.linalg.norm(x[6:9])-limit,0)]])
   return np.concatenate(residual)
  fit=least_squares(fun,np.clip(initial,low+1e-8,high-1e-8),bounds=(low,high),max_nfev=65,ftol=1e-9,xtol=1e-9,gtol=1e-8)
  rot,p=unpack(fit.x);surface,dist,pen,vertices=evaluate(q,rot,p,fit.x[:6]);q=d.qpos[:23].copy();prev=fit.x[:6].copy()
  if active:prev_r=fit.x[6:9].copy();prev_delta=fit.x[9:11].copy()
  qout.append(q);pout.append(p);rout.append(rot.as_quat());stats.append([float(fit.cost),*dist,*pen,np.degrees((rot.inv()*fullR[k]).magnitude())])
  if k%40==0:print('object solve',k,'gap',dist,'penetration',pen,'rotation error deg',stats[-1][-1],flush=True)
 qout=np.array(qout);pout=np.array(pout);rotations=R.from_quat(rout);q=q0.copy();q[:,7+fi]=PchipInterpolator(ts,qout[:,7+fi])(st);dense_rot=Slerp(ts,rotations)(st);pos=PchipInterpolator(ts,pout)(st)
 # Correct interpolated z with exact oriented support; every frame has correct clearance.
 clearance=PchipInterpolator(ts,prior['source_clearance'])(st)*cfg['clearance_scale'];pos[:,2]=np.abs(dense_rot.as_matrix()[:,2,:])@half+clearance
 # Dense front-only cleanup. Object trajectory stays fixed; rear wheels remain untouched.
 errors=[];prev=q[0,7+fi].copy()
 for k in range(N):
  rot=dense_rot[k];p=pos[k];contact_on=z['grip_blend'][k];gap=z['jaw_gap'][k];withdraw=max(z['contact_targets'][k,0,2]-z['object_pos'][k,2]-half[2]-.04,0);target=np.array([[-.09,half[1]+gap,half[2]+.04+withdraw],[-.09,-half[1]-gap,half[2]+.04+withdraw]])
  def fun(x):
   surface,dist,pen,_=evaluate(q[k],rot,p,x);return np.r_[(surface-target).ravel()*.2,dist*5*contact_on,100*np.minimum(dist+1e-5,0),100*np.maximum(pen-1e-5,0),.0005*(x-prev)]
  fit=least_squares(fun,q[k,7+fi],bounds=(limits[fi,0]+1e-6,limits[fi,1]-1e-6),max_nfev=25,ftol=1e-9,gtol=1e-8,xtol=1e-9);prev=fit.x;surface,dist,pen,_=evaluate(q[k],rot,p,prev);q[k,7+fi]=prev;errors.append([*dist,*pen])
  if k%800==0:print('dense contacts',k,'gap',dist,'penetration',pen,flush=True)
 # Robot and object velocities are derived from the final corrected trajectories.
 import sys
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/resmimic_rolling_v2'))
 from build_reference import velocities,audit
 rm=mujoco.MjModel.from_xml_path(cfg['robot_xml']);v=velocities(rm,q,1/cfg['output_fps']);a=np.gradient(v,1/cfg['output_fps'],axis=0)
 full_dense=HR*Slerp(ts,relative)(st);baselineR=R.from_quat(z['object_quat'][:,[1,2,3,0]]);err_old=np.degrees((baselineR.inv()*full_dense).magnitude());err_new=np.degrees((dense_rot.inv()*full_dense).magnitude());contact=(st>=34/30)&(st<=153/30)
 z.update(qpos=q,qvel=v,qacc=a,root_pos=q[:,:3],root_rot=q[:,[4,5,6,3]],dof_pos=q[:,7:],object_pos=pos,object_quat=dense_rot.as_quat()[:,[3,0,1,2]],object_lin_vel=np.gradient(pos,1/cfg['output_fps'],axis=0),object_prior_quat=full_dense.as_quat()[:,[3,0,1,2]],object_clearance=clearance,wheel_metrics=audit(rm,q,cfg['output_fps']),object_half_size=half,object_mass=cfg['box_mass_kg'],training_ready=False,payload_manipulation_validated=False)
 # Old contact_targets are obsolete after object rotation; store the actual new semantic proxy targets.
 z['contact_targets']=np.array([pos[k]+dense_rot[k].apply(np.array([[-.09,half[1]+z['jaw_gap'][k],half[2]+.04],[-.09,-half[1]-z['jaw_gap'][k],half[2]+.04]])) for k in range(N)])
 np.savez(out/'reference.npz',**z);np.savez(out/'optimization.npz',source_qpos=qout,source_object_pos=pout,source_object_quat=rotations.as_quat()[:,[3,0,1,2]],source_stats=stats,dense_errors=errors,orientation_error_before_deg=err_old,orientation_error_after_deg=err_new)
 report=dict(method='object-frame interaction Laplacian + bilateral surface/gap + collision penalties + SE3 prior; independent hard validation required',full_rotation_prior=True,rotation_limit_deg=cfg['object_rotation_limit_deg'],rotation_error_before_carry_rms_deg=float(np.sqrt(np.mean(err_old[contact]**2))),rotation_error_after_carry_rms_deg=float(np.sqrt(np.mean(err_new[contact]**2))),rear_qpos_unchanged=bool(np.array_equal(q[:,15:],q0[:,15:])),root_unchanged=bool(np.array_equal(q[:,:7],q0[:,:7])),solver='scipy least_squares; not the original SQP solver',dynamics_validated=False)
 (out/'retarget_report.json').write_text(json.dumps(report,indent=2));return report
if __name__=='__main__':
 import sys
 c=json.loads(Path(sys.argv[1]).read_text());print(run(c,c['output'],'experiments/human_ground_clamp/reference.npz'))
