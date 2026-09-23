"""One-clip kinematic study, not a dynamically validated policy or official GMR retarget.
Source: ResMimic's already-retargeted G1 carry.pkl and paired object carry.npz.
Outputs remain separate from both source repositories. Joint arrays use MuJoCo order.
"""
from pathlib import Path
import json, pickle
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio

OUT = Path(__file__).resolve().parent
SRC = Path('/home/robotennis2025/ResMimic/legged_gym/assets/motions')
MODEL = Path('/home/robotennis2025/SJTU-BipedGo2w/GMR/assets/unitree_go2w/go2w.xml')
s = pickle.load(open(SRC/'carry.pkl','rb'))
o = np.load(SRC/'carry.npz')
fps, n = s['fps'], len(s['root_pos'])
names = s['link_body_list']
local = s['local_body_pos']
rot = R.from_quat(s['root_rot'])
world = np.einsum('nij,nkj->nki',rot.as_matrix(),local)+s['root_pos'][:,None,:]
yaw = np.unwrap(rot.as_euler('xyz')[:,2])
heading = R.from_euler('z',yaw).as_matrix()
# Work in heading-only coordinates so source pelvis pitch does not become dog tilt.
relative = np.einsum('nji,nkj->nki',heading,world-s['root_pos'][:,None,:])
get = lambda name: relative[:,names.index(name)]
# Source hands are on negative local X in this supplied clip. Reflect sagittal
# reach for this morphology, keeping left/right sides and source time variation.
hands = np.stack([get('left_rubber_hand'),get('right_rubber_hand')],axis=1)*.75
hands[:,:,0] *= -1
hands[:,:,0] += .08
hands[:,:,1] *= 1.5
hands[:,:,2] += .03
feet = np.stack([get('left_ankle_roll_link'),get('right_ankle_roll_link')],axis=1)*.65
feet[:,:,0] -= np.mean(feet[120:,:,0])
feet[:,:,1] *= 1.8
height = .55 + .65*(s['root_pos'][:,2]-np.median(s['root_pos'][140:,2]))
feet[:,:,2] = .086-height[:,None]+.65*(world[:,[names.index('left_ankle_roll_link'),names.index('right_ankle_roll_link')],2]-.04)
# Hand-selected provisional phase, based on this clip's lift and travel curves.
# Freeze the local support/hold geometry after a smooth 110..135 transition.
start,end = 110,135
blend = np.clip((np.arange(n)-start)/(end-start),0,1)
blend = blend**3*(10-15*blend+6*blend**2)
hold_hands = np.median(hands[140:210],axis=0)
fixed_feet = np.array([[0,.15,.086-.55],[0,-.15,.086-.55]])
base_targets=np.concatenate([hands,feet],axis=1)
adapt_targets=base_targets*(1-blend[:,None,None])+np.concatenate([hold_hands,fixed_feet])[None]*blend[:,None,None]
adapt_height=height*(1-blend)+.55*blend
m=mujoco.MjModel.from_xml_path(str(MODEL)); d=mujoco.MjData(m)
leg_idx=np.array([i for i in range(16) if i%4!=3]); wheel_idx=np.array([3,7,11,15])
limits=m.jnt_range[1:][leg_idx]
ids=[m.body(x).id for x in ['FL_wheel_link','FR_wheel_link','RL_wheel_link','RR_wheel_link']]
upright=R.from_euler('y',-90,degrees=True)
# all body centers for rendering/diagnostics
bodies=[m.body(i).name for i in range(1,m.nbody)]

def solve(targets, heights, tag):
 seed=np.tile([0,.8,-1.5],4); qseq=[]; residual=[]
 d.qpos[3:7]=upright.as_quat()[[3,0,1,2]]
 for f in range(n):
  d.qpos[:3]=[0,0,heights[f]]
  target=targets[f]+d.qpos[:3]
  prev=seed.copy()
  def fun(q):
   d.qpos[7+leg_idx]=q;mujoco.mj_forward(m,d)
   e=(d.xpos[ids]-target)*np.array([2,2,5,5])[:,None]
   return np.r_[e.ravel(),.012*(q-prev)]
  res=least_squares(fun,seed,bounds=(limits[:,0]+1e-5,limits[:,1]-1e-5),max_nfev=90,ftol=1e-7)
  seed=res.x; fun(seed)
  residual.append(np.linalg.norm(d.xpos[ids]-target,axis=1))
  q=d.qpos.copy(); q[:2]=s['root_pos'][f,:2]-s['root_pos'][0,:2]
  q[3:7]=(R.from_euler('z',yaw[f])*upright).as_quat()[[3,0,1,2]]
  qseq.append(q)
 qseq=np.array(qseq)
 # Rear wheel rolling angles from projected hub displacement, radius assumption
 # checked against the mesh below. Front wheel angles remain zero.
 positions=[]
 for q in qseq:
  d.qpos[:]=q;mujoco.mj_forward(m,d);positions.append(d.xpos[1:].copy())
 positions=np.array(positions)
 for j,b in [(11,'RL_wheel_link'),(15,'RR_wheel_link')]:
  p=positions[:,bodies.index(b)]; delta=np.diff(p,axis=0)
  forward=np.c_[np.cos(yaw[1:]),np.sin(yaw[1:]),np.zeros(n-1)]
  # sign follows local joint +Y axis in model
  qseq[:,7+j]=np.r_[0,np.cumsum(np.sum(delta*forward,axis=1)/.086)]
 for f,q in enumerate(qseq):
  d.qpos[:]=q;mujoco.mj_forward(m,d);positions[f]=d.xpos[1:]
 np.savez(OUT/f'{tag}.npz',fps=fps,qpos=qseq,root_pos=qseq[:,:3],root_rot=qseq[:,[4,5,6,3]],dof_pos=qseq[:,7:],body_pos_w=positions,body_names=np.array(bodies),joint_names=np.array([m.joint(i).name for i in range(1,m.njnt)]),carry_blend=blend,target_error_m=np.array(residual))
 with open(OUT/f'{tag}.pkl','wb') as h:pickle.dump(dict(fps=fps,root_pos=qseq[:,:3],root_rot=qseq[:,[4,5,6,3]],dof_pos=qseq[:,7:],local_body_pos=None,link_body_list=None),h)
 return qseq,positions,np.array(residual)

bq,bp,be=solve(base_targets,height,'go2w_framewise')
aq,ap,ae=solve(adapt_targets,adapt_height,'go2w_carry_adapted')
# Basic integrity and the intended transport-phase invariant.
assert np.isfinite(aq).all()
assert np.all(aq[:,7+leg_idx]>=limits[:,0]-1e-6) and np.all(aq[:,7+leg_idx]<=limits[:,1]+1e-6)
print('max carry leg range',np.ptp(aq[160:,7+leg_idx],axis=0).max(),flush=True)
summary=dict(source=str(SRC/'carry.pkl'),paired_object=str(SRC/'carry.npz'),source_kind='G1 retargeted motion, not raw human SMPL-X',frames=n,fps=fps,duration_s=(n-1)/fps,phase_transition_frames_zero_based=[start,end],phase_transition_s=[start/fps,end/fps],phase_selection='provisional manual selection for this one clip; no release segment observed',method='position-only bounded IK prototype; not existing GMR baseline',sagittal_hand_reflection=True,mesh_based_wheel_radius_m=.086,mean_target_error_m=ae.mean(axis=0).tolist(),max_target_error_m=ae.max(axis=0).tolist(),carry_leg_peak_to_peak_rad=float(np.ptp(aq[160:,7+leg_idx],axis=0).max()),root_xy_path_length_m=float(np.linalg.norm(np.diff(aq[:,:2],axis=0),axis=1).sum()),limitations=['kinematics only; no contact-force, balance, collision or policy validation','source is G1 proxy; morphology transfer cannot recover original human motion','phase-specific mapping and dimensions are heuristic','rear wheel forward rolling approximation; lateral slip not constrained','NPZ is an inspection format, not directly Isaac Lab training input'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
# Source skeleton uses actual supplied link positions, connected via source URDF.
import xml.etree.ElementTree as ET
urdf=ET.parse('/home/robotennis2025/ResMimic/assets/g1/g1_custom_collision_29dof.urdf')
source_edges=[]
for j in urdf.findall('joint'):
 a=j.find('parent').get('link');b=j.find('child').get('link')
 if a in names and b in names:source_edges.append((names.index(a),names.index(b)))
dog_edges=[(m.body_parentid[i]-1,i-1) for i in range(1,m.nbody) if m.body_parentid[i]>0]
source_world=world-s['root_pos'][0]*np.array([1,1,0])
fig=plt.figure(figsize=(13,4.8));axes=[fig.add_subplot(1,3,i+1,projection='3d') for i in range(3)]
def draw(f):
 for ax in axes:ax.clear()
 for ax,p,edges,title,col in zip(axes,[source_world[f],bp[f],ap[f]],[source_edges,dog_edges,dog_edges],['Source: G1 carry','Go2W: framewise position IK','Go2W: adapted carry'],['#555555','#c68127','#157a8b']):
  for a,b in edges:ax.plot(*p[[a,b]].T,color=col,lw=3)
  ax.scatter(*p.T,color=col,s=6)
  if title.startswith('Go2W'):
   theta=np.linspace(0,2*np.pi,32)
   for name in ['FL_wheel_link','FR_wheel_link','RL_wheel_link','RR_wheel_link']:
    c=p[bodies.index(name)]
    ring=c+np.c_[.086*np.cos(theta)*np.cos(yaw[f]),.086*np.cos(theta)*np.sin(yaw[f]),.086*np.sin(theta)]
    ax.plot(*ring.T,color=col,lw=1.3)
  ax.set_xticks([-.5,0,.5,1]);ax.set_yticks([-1,-.5,0]);ax.set_zticks([0,.4,.8,1.2])
  ax.set(xlim=(-.6,1.25),ylim=(-1.3,.5),zlim=(0,1.35),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title=title)
  ax.set_box_aspect((1.85,1.8,1.35));ax.view_init(elev=19,azim=35)
  xx,yy=np.meshgrid([-.6,1.25],[-1.3,.5]);ax.plot_surface(xx,yy,xx*0,alpha=.08,color='gray')
 phase='PICK / LIFT' if f<start else 'BLEND' if f<end else 'HOLD / TRANSPORT'
 fig.suptitle(f'ResMimic carry | frame {f:03d} | {f/fps:.2f} s | {phase}\nKinematic prototype only; no physics / grasp validation',fontsize=12)
 fig.tight_layout(rect=[0,0,1,.88]);fig.canvas.draw()
 return np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
frames=[]
for f in range(0,n,3):
 frames.append(draw(f))
 if f==174:fig.savefig(OUT/'comparison.png',dpi=150)
imageio.mimsave(OUT/'comparison.gif',frames,duration=100,loop=0)
print(json.dumps(summary,indent=2),flush=True)
