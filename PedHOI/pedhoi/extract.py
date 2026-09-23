"""Run with the configured SMPL-X Python environment."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,joblib,torch,smplx

def run(cfg,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);records=joblib.load(cfg['raw_omomo']);items=records.items() if isinstance(records,dict) else enumerate(records);matches=[(i,v) for i,v in items if v['seq_name']==cfg['sequence']]
 if len(matches)!=1:raise ValueError(f"Expected one exact sequence, found {len(matches)}")
 idx,e=matches[0];torch.set_num_threads(4);model=smplx.create(cfg['smplx_models'],model_type='smplx',gender=str(e['gender'].item()),ext='npz',num_betas=16,use_pca=False);joints=[]
 with torch.no_grad():
  for a in range(0,len(e['trans']),32):
   b=min(a+32,len(e['trans']));n=b-a;z=model(global_orient=torch.tensor(e['root_orient'][a:b],dtype=torch.float32),body_pose=torch.tensor(e['pose_body'][a:b].reshape(n,63),dtype=torch.float32),transl=torch.tensor(e['trans'][a:b],dtype=torch.float32),betas=torch.tensor(e['betas'],dtype=torch.float32).expand(n,-1),left_hand_pose=torch.zeros(n,45),right_hand_pose=torch.zeros(n,45),jaw_pose=torch.zeros(n,3),leye_pose=torch.zeros(n,3),reye_pose=torch.zeros(n,3),expression=torch.zeros(n,10));joints.append(z.joints[:,:22].numpy())
 j=np.concatenate(joints);obj=e['obj_com_pos'];height=obj[:,2];base=np.median(np.r_[height[:10],height[-10:]]);lift=np.flatnonzero(height>base+.15);carry=np.flatnonzero(height>base+.65*(height.max()-base))
 np.savez(out/'human_source.npz',fps=cfg['source_fps'],joints_w=j,root_orient=e['root_orient'],object_pos=obj,object_rot=e['obj_rot'],object_scale=e['obj_scale'],object_trans=e['obj_trans'],pose_body=e['pose_body'],trans=e['trans'],betas=e['betas'],gender=e['gender'])
 meta=dict(sequence=cfg['sequence'],source_file=cfg['raw_omomo'],entry_index=idx,source_sha256=hashlib.sha256(Path(cfg['raw_omomo']).read_bytes()).hexdigest(),frames=len(j),fps=cfg['source_fps'],phases_zero_based=dict(lift_start=int(lift[0]),carry_start=int(carry[0]),carry_end=int(carry[-1]),place_end=int(lift[-1]),end=len(j)-1),phase_method='coarse bootstrap thresholds only; object interaction stage has explicit contact/ground semantics',units='meters, world Z-up',body_model='SMPL-X first 22 anatomical joints; no G1 intermediary')
 (out/'source.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':run(json.loads(Path(sys.argv[1]).read_text()),sys.argv[2])
