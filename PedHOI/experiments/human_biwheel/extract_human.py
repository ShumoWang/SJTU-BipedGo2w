from pathlib import Path
import hashlib,json
import joblib,numpy as np,torch,smplx
P=Path(__file__).resolve().parent
source=Path('/home/robotennis2025/G2E-HOI/data/00_omomo_raw/official_omomo/data/test_diffusion_manip_seq_joints24.p')
e=joblib.load(source)[41];assert e['seq_name']=='sub16_largebox_010'
torch.set_num_threads(4)
model=smplx.create('/home/robotennis2025/GMR/assets/body_models',model_type='smplx',gender=str(e['gender'].item()),ext='npz',num_betas=16,use_pca=False)
T=len(e['trans']);joints=[]
with torch.no_grad():
 for a in range(0,T,32):
  b=min(a+32,T);n=b-a
  out=model(global_orient=torch.tensor(e['root_orient'][a:b],dtype=torch.float32),body_pose=torch.tensor(e['pose_body'][a:b].reshape(n,63),dtype=torch.float32),transl=torch.tensor(e['trans'][a:b],dtype=torch.float32),betas=torch.tensor(e['betas'],dtype=torch.float32).expand(n,-1),left_hand_pose=torch.zeros(n,45),right_hand_pose=torch.zeros(n,45),jaw_pose=torch.zeros(n,3),leye_pose=torch.zeros(n,3),reye_pose=torch.zeros(n,3),expression=torch.zeros(n,10))
  joints.append(out.joints[:,:22].numpy())
joints=np.concatenate(joints)
obj=e['obj_com_pos'];z=obj[:,2];base=np.median(np.r_[z[:10],z[-10:]])
lift=np.where(z>base+.15)[0];plateau=np.where(z>base+.65*(z.max()-base))[0]
phases=dict(lift_start=int(lift[0]),carry_start=int(plateau[0]),carry_end=int(plateau[-1]),place_end=int(lift[-1]),end=T-1)
np.savez(P/'human_source.npz',fps=30,joints_w=joints,root_orient=e['root_orient'],object_pos=obj,object_rot=e['obj_rot'],pose_body=e['pose_body'],trans=e['trans'],betas=e['betas'],gender=e['gender'])
meta=dict(sequence=e['seq_name'],dataset='OMOMO official test',source_file=str(source),entry_index=41,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),frames=T,fps=30,source_duration_s=(T-1)/30,body_model='local SMPL-X, first 22 anatomical joints, no G1 intermediate',phases_zero_based=phases,phase_method='object height threshold: 0.15 m lift, 65% peak height plateau; heuristic, not contact labels',object_height_start_peak_end=[float(z[0]),float(z.max()),float(z[-1])],body_z_range=[float(joints[:,:,2].min()),float(joints[:,:,2].max())],units='meters, native OMOMO world Z-up',limitations=['object contact labels inferred only','no claim of physical grasp forces'])
(P/'source.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
