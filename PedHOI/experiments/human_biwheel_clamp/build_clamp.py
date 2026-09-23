"""Object-centred two-front-wheel support retarget. Kinematics, not a grasp claim."""
from pathlib import Path
import sys,json,xml.etree.ElementTree as ET
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
from scipy.integrate import cumulative_trapezoid
P=Path(__file__).resolve().parent;OLD=P.parent/'human_biwheel'
sys.path.insert(0,str(P.parent/'resmimic_rolling_v2'))
from build_reference import velocities,audit,MODEL
m=mujoco.MjModel.from_xml_path(MODEL);d=mujoco.MjData(m)
z=dict(np.load(OLD/'reference.npz'));q=z['qpos'].copy();tt=z['time'];N=len(q)
h=np.load(OLD/'human_source.npz');src=z['source_time'];meta=json.loads((OLD/'source.json').read_text());ph=meta['phases_zero_based']
# 24 x 44 x 18 cm, lightweight 0.5 kg test box. Dimensions are retargeted.
size=np.array([.10,.16,.05]);mass=.5
fw=R.from_quat(q[:,[4,5,6,3]]).apply(np.tile([0,0,-1.],(N,1)));yaw=np.unwrap(np.arctan2(fw[:,1],fw[:,0]));H=R.from_euler('z',yaw)
# Back away from the pickup support before turning, still differential-drive.
phase=np.clip((tt-5.2)/1.6,0,1);extra=-.42*np.sin(np.pi*phase)**2;extra[(tt<=5.2)|(tt>=6.8)]=0
delta=cumulative_trapezoid(extra[:,None]*fw,tt,axis=0,initial=0);q[:,:3]+=delta
wheel_extra=cumulative_trapezoid(extra/.086,tt,initial=0);q[:,18]+=wheel_extra;q[:,22]+=wheel_extra
z['forward_speed']=z['forward_speed']+extra;z['axle_xy']=z['axle_xy']+delta[:,:2]
zobj=h['object_pos'][:,2];lift=np.clip((zobj-np.mean(zobj[:30]))/(np.median(zobj[60:125])-np.mean(zobj[:30])),0,1)
# Preserve lift/lower timing, with a flat transport plateau.
lift[ph['carry_start']:ph['carry_end']+1]=1

def smooth(x):
 x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
# Lift before transport; lower only after the rolling path reaches the destination.
lift_start=ph['lift_start']/30*4
lift_end=ph['carry_start']/30*4
place_start=18.8;place_end=21.2
height=.63+.12*smooth((tt-lift_start)/(lift_end-lift_start))*(1-smooth((tt-place_start)/(place_end-place_start)))
box=q[:,:3]+H.apply(np.tile([.25,0,0.],(N,1)));box[:,2]=height
pick=ph['lift_start']*16;place=int(round(place_end*120))
box[:pick]=box[pick];box[place:]=box[place];byaw=yaw.copy();byaw[:pick]=yaw[pick];byaw[place:]=yaw[place]
def gate(x):x=np.clip(x,0,1);return x*x*x*(10-15*x+6*x*x)
# Approach from below, establish contact before lift; lower away after placement.
gap=.045*(1-gate((src-(ph['lift_start']-18)/30)/(16/30)))+.045*gate((tt-(place_end+.2))/2.)
fi=np.array([0,1,2,4,5,6]);limits=m.jnt_range[1:];seed=q[0,7+fi].copy();gids=[];verts=[]
for name in ['FL_wheel_link','FR_wheel_link']:
 g=np.where(m.geom_bodyid==m.body(name).id)[0][0];gids.append(g);mid=m.geom_dataid[g];verts.append(m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]].copy())
def contacts():
 out=[]
 for g,v in zip(gids,verts):
  world=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
  local=(world-box[k])@bh.as_matrix()
  center=(local.max(0)+local.min(0))/2
  center[1]=local[:,1].min() if len(out)==0 else local[:,1].max()
  out.append(box[k]+bh.apply(center))
 return np.array(out)
errors=[];targets=[]
for k in range(N):
 d.qpos[:]=q[k];bh=R.from_euler('z',byaw[k]);tgt=box[k]+bh.apply(np.array([[0,size[1]+gap[k],-.08],[0,-size[1]-gap[k],-.08]]))
 def fun(x):
  d.qpos[7+fi]=x;mujoco.mj_forward(m,d)
  return np.r_[(contacts()-tgt).ravel(),.0001*(x-seed)]
 fit=least_squares(fun,seed,bounds=(limits[fi,0]+1e-6,limits[fi,1]-1e-6),max_nfev=65,gtol=1e-10,ftol=1e-10,xtol=1e-10);seed=fit.x;fun(seed);q[k]=d.qpos;errors.append(np.linalg.norm(contacts()-tgt,axis=1));targets.append(tgt)
v=velocities(m,q,1/float(z['fps']));a=np.gradient(v,1/float(z['fps']),axis=0)
z.update(qpos=q,qvel=v,qacc=a,root_pos=q[:,:3],root_rot=q[:,[4,5,6,3]],dof_pos=q[:,7:],object_pos=box,object_quat=R.from_euler('z',byaw).as_quat()[:,[3,0,1,2]],object_half_size=size,object_mass=mass,contact_targets=np.array(targets),support_gap=gap)
for stale in ['front_targets_local','source_front_ik_error']:
 z.pop(stale,None)
z.update(training_ready=False,payload_manipulation_validated=False)
np.savez(P/'reference.npz',**z)
# Explicit free box and two fixed support tables. No welds/mocap/equality constraints.
tree=ET.parse(MODEL);root=tree.getroot();root.find('compiler').set('meshdir',str(Path(MODEL).parent/'assets'));world=root.find('worldbody')
b=ET.SubElement(world,'body',name='payload',pos=' '.join(map(str,box[0])));ET.SubElement(b,'freejoint',name='payload_free');ET.SubElement(b,'geom',name='payload_box',type='box',size=' '.join(map(str,size)),mass=str(mass),rgba='.65 .36 .13 1',friction='.8 .02 .01',condim='6')
for label,k in [('pick',pick),('place',place)]:
 # Narrow central pedestal leaves the front wheels free below either side.
 pos=box[k].copy();pos[2]=box[k,2]-size[2]-.015
 b=ET.SubElement(world,'body',name=label+'_table',pos=' '.join(map(str,pos)),quat=' '.join(map(str,z['object_quat'][k])))
 ET.SubElement(b,'geom',name=label+'_table_geom',type='box',size='.060 .060 .015',rgba='.45 .5 .55 1',friction='.8 .02 .01')
tree.write(P/'scene.xml')
mm=mujoco.MjModel.from_xml_path(str(P/'scene.xml'));dd=mujoco.MjData(mm);obj=mm.geom('payload_box').id;penetration=0.;other_pen=0.;dist=[];bad=[]
for k in range(N):
 dd.qpos[:23]=q[k];dd.qpos[23:26]=box[k];dd.qpos[26:30]=z['object_quat'][k];mujoco.mj_forward(mm,dd)
 dist.append([mujoco.mj_geomDistance(mm,dd,int(g),obj,1.,None) for g in gids])
 for c in dd.contact:
  b1,b2=mm.geom_bodyid[c.geom1],mm.geom_bodyid[c.geom2]
  if obj in [c.geom1,c.geom2]:
   other=c.geom2 if c.geom1==obj else c.geom1
   if mm.geom_bodyid[other]<m.nbody and mm.geom_bodyid[other]>0 and other not in gids and other not in [g+1 for g in gids]:
    penetration=max(penetration,-c.dist)
  if c.dist<0 and b1!=0 and b2!=0 and obj not in [c.geom1,c.geom2]:other_pen=max(other_pen,-c.dist)
metrics=audit(m,q,float(z['fps']));carry=(src>=ph['carry_start']/30)&(src<=ph['carry_end']/30);dist=np.array(dist)
z['wheel_metrics']=metrics;np.savez(P/'reference.npz',**z)
report=dict(scope='object-centred side-clamp reference; free-payload dynamics checked separately',box_dimensions_m=(size*2).tolist(),assumed_box_mass_kg=mass,support='opposing inner wheel faces clamp lower sides of box; calf hubs below box',front_surface_fit_max_m=float(np.max(errors)),carry_wheel_box_distance_max_abs_m=float(np.max(np.abs(dist[carry]))),box_other_robot_penetration_max_m=penetration,non_box_non_floor_penetration_max_m=other_pen,rear_no_slip_max_m_s=float(metrics[:,:,2].max()),rear_leg_angles_unchanged=bool(np.array_equal(q[:,[15,16,17,19,20,21]],np.load(OLD/'reference.npz')['qpos'][:,[15,16,17,19,20,21]])),pickup_reverse_clearance_m=float(np.linalg.norm(delta[-1,:2])),dynamics_validated=False)
np.savez(P/'geometry_check.npz',surface_errors=errors,wheel_box_distance=dist,wheel_metrics=metrics)
report.update(geometry_pass=bool(np.max(errors)<.001 and penetration<.001 and other_pen<.001 and metrics[:,:,2].max()<.003),payload_manipulation_validated=False,reference_only=True,pick_place_supports='fixed thin shelves',placement_end_s=place_end)
(P/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
