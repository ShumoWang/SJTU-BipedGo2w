"""OmniRetarget-style object-frame interaction mesh, with frame audit.
Independent implementation of the published Delaunay + uniform Laplacian idea;
not a claim of running the original G1 solver or transferring its certification.
"""
from pathlib import Path
import json,itertools
import numpy as np,trimesh
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R,Slerp

def corners(half):return np.array(list(itertools.product([-1,1],repeat=3)))*half

def laplacian(points):
 simplices=Delaunay(points).simplices;adj=[set() for _ in points]
 for tet in simplices:
  for i in tet:adj[i].update(set(tet)-{i})
 L=np.eye(len(points))
 for i,neighbors in enumerate(adj):
  if neighbors:L[i,list(neighbors)]=-1/len(neighbors)
 return L

def run(cfg,out,human,seed):
 out=Path(out);h=np.load(human);j=h['joints_w'];raw=R.from_matrix(h['object_rot']);omni=np.load(cfg['omni_reference']);oq=omni['qpos']
 if oq.shape!=(len(j),43) or float(omni['fps'])!=cfg['source_fps']:raise ValueError('Omni reference sequence/time schema mismatch')
 # Official file is Drake [quat(wxyz), xyz, joints29, object quat(wxyz), xyz].
 orot=R.from_quat(oq[:,[37,38,39,36]]);world_change=orot*raw.inv();gauge=world_change.mean();gauge_errors=np.degrees((world_change*gauge.inv()).magnitude())
 if gauge_errors.max()>.01:raise ValueError('Omni and raw object rotations are not consistently frame-aligned')
 # Choose material axes from object OBB, with +Y toward the human left wrist.
 mesh=trimesh.load(cfg['omni_mesh'],force='mesh');axes=mesh.bounding_box_oriented.primitive.transform[:3,:3].copy()
 span=np.einsum('ni,nij->nj',j[34:154,20]-j[34:154,21],h['object_rot'][34:154]);med=np.median(span,axis=0)
 width=axes[:,np.argmax(np.abs(med@axes))].copy();width*=np.sign(med@width)
 up=axes[:,np.argmax(np.abs(raw[0].as_matrix()[2]@axes))].copy();up*=np.sign(raw[0].apply(up)[2]);forward=np.cross(width,up)
 initial=raw[0].as_matrix()@np.column_stack([forward,width,up]);yaw0=np.arctan2(initial[1,0],initial[0,0]);basis=raw[0].inv()*R.from_euler('z',yaw0)
 objrot=raw*basis;canonical_vertices=np.asarray(mesh.vertices)@basis.as_matrix();source_half=np.ptp(canonical_vertices,axis=0)/2
 target_half=np.array(cfg['box_dimensions_m'])/2;source_corners=corners(source_half);target_corners=corners(target_half)
 side=j[:,1]-j[:,2];side[:,2]=0;side/=np.linalg.norm(side,axis=1)[:,None];fwd=np.cross(side,[0,0,1]);hyaw=np.unwrap(np.arctan2(fwd[:,1],fwd[:,0]));relative=R.from_euler('z',-hyaw)*objrot
 mapped=[];Ls=[];targets=[];handanchors=[]
 for k in range(len(j)):
  points=objrot[k].inv().apply(j[k,[0,12,18,19,20,21]]-h['object_pos'][k]);handanchors.append(points[-2:])
  source=np.vstack([points,source_corners]);L=laplacian(source);morph=source*(target_half/source_half);mapped.append(morph);Ls.append(L);targets.append(L@morph)
 support=np.abs(objrot.as_matrix()[:,2,:])@source_half
 clearance=h['object_pos'][:,2]-support;floor=np.median(np.r_[clearance[:25],clearance[-25:]])
 clearance=np.maximum(clearance-floor,0);clearance[:32]=0;clearance[156:]=0
 np.savez(out/'interaction.npz',source_object_quat=objrot.as_quat()[:,[3,0,1,2]],source_object_relative_quat=relative.as_quat()[:,[3,0,1,2]],canonical_basis=basis.as_matrix(),omni_world_alignment=gauge.as_matrix(),source_box_half_size=source_half,target_box_half_size=target_half,source_hand_anchors_local=handanchors,source_vertices=mapped,laplacian=Ls,target_laplacian=targets,source_clearance=clearance,source_corners=source_corners,target_corners=target_corners)
 report=dict(sequence=cfg['sequence'],method='object-frame Delaunay interaction graph + uniform Laplacian',upstream='amazon-far/holosoma interaction_mesh_retargeter.py and src/utils.py',omni_role='object-frame/time audit, not G1 robot intermediary',omni_object_rotation_alignment_max_deg=float(gauge_errors.max()),omni_world_yaw_deg=float(gauge.as_euler('xyz',degrees=True)[2]),quaternion_layout='wxyz on disk; scipy xyzw explicitly converted',source_box_half_size_m=source_half.tolist(),target_box_half_size_m=target_half.tolist(),ground_height='oriented shape support, not centre_z minus fixed half-height',contacts_inferred=True,contact_annotation='heuristic episode-specific grip window; no ground-truth contact labels',full_original_rotation_is_prior=True)
 (out/'object_prior_report.json').write_text(json.dumps(report,indent=2));return report
if __name__=='__main__':
 import sys
 c=json.loads(Path(sys.argv[1]).read_text());out=Path(c['output']);out.mkdir(parents=True,exist_ok=True);print(run(c,out,'experiments/human_biwheel/human_source.npz','experiments/human_ground_clamp/reference.npz'))
