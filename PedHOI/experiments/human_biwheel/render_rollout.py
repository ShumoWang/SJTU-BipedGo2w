import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
from scipy.interpolate import PchipInterpolator
import imageio.v2 as io
P=Path(__file__).resolve().parent
m=mujoco.MjModel.from_xml_path('/home/robotennis2025/SJTU-BipedGo2w/GMR/assets/unitree_go2w/go2w.xml');d=mujoco.MjData(m)
m.vis.headlight.ambient[:]=.6;m.vis.headlight.diffuse[:]=.8;m.vis.global_.offwidth=640;m.vis.global_.offheight=480
r=mujoco.Renderer(m,height=480,width=640);cam=mujoco.MjvCamera();cam.distance=2.;cam.elevation=-15;cam.azimuth=25
z=np.load(P/'reference.npz');qr=PchipInterpolator(z['time'],z['qpos']);rows=np.load(P/'balance_tracking_rollout.npz')['records'];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);preview=[]
with io.get_writer(P/'physics_rollout.mp4',fps=10,macro_block_size=1) as writer:
 for k in range(0,len(rows),10):
  row=rows[k];t=row[0];ref=qr(t);ref[3:7]/=np.linalg.norm(ref[3:7]);sim=row[2:25];cam.lookat[:]=ref[:3]
  out=Image.new('RGB',(1280,546),'white');draw=ImageDraw.Draw(out)
  for side,q in enumerate([ref,sim]):
   d.qpos[:]=q;mujoco.mj_forward(m,d);r.update_scene(d,camera=cam);out.paste(Image.fromarray(r.render()),(side*640,66))
  phase='PICK' if t<5.2 else 'LIFT' if t<6.8 else 'CARRY' if t<17.47 else 'PLACE' if t<19.33 else 'RETRACT'
  draw.text((15,5),'Reference (kinematics)',fill='black',font=font);draw.text((655,5),'Actual torque-controlled physics (unloaded)',fill='black',font=font)
  draw.text((15,32),f'{phase} | robot time {t:.2f}s | root error {row[1]*100:.1f}cm | no root support / no pose resets',fill='#555555',font=font)
  writer.append_data(np.asarray(out))
  if k%40==0:preview.append(np.asarray(out))
  if k==1200:out.save(P/'physics_keyframe.png')
io.mimsave(P/'physics_preview_4x.gif',preview,duration=100,loop=0)
r.close();print('Saved full-speed actual physics video and explicitly 4x preview')
