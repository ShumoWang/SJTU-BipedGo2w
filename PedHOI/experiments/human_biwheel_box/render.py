import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as io
from scipy.spatial.transform import Rotation as R
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parent;z=np.load(P/'reference.npz');m=mujoco.MjModel.from_xml_path(str(P/'scene.xml'));d=mujoco.MjData(m)
m.vis.headlight.ambient[:]=.65;m.vis.global_.offwidth=640;m.vis.global_.offheight=480
r=mujoco.Renderer(m,480,640);cam=mujoco.MjvCamera();cam.distance=1.8;cam.elevation=-22
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);frames=[]
with io.get_writer(P/'box_reference.mp4',fps=30,macro_block_size=1) as writer:
 for k in range(0,len(z['qpos']),8):
  d.qpos[:23]=z['qpos'][k];d.qpos[23:26]=z['object_pos'][k];d.qpos[26:]=z['object_quat'][k];mujoco.mj_forward(m,d)
  f=R.from_quat(d.qpos[[4,5,6,3]]).apply([0,0,-1]);yaw=np.degrees(np.arctan2(f[1],f[0]));cam.lookat[:]=(d.qpos[:3]+d.qpos[23:26])/2
  out=Image.new('RGB',(1280,546),'white');draw=ImageDraw.Draw(out)
  for s,ang in enumerate([95,180]):
   cam.azimuth=yaw+ang;r.update_scene(d,camera=cam);out.paste(Image.fromarray(r.render()),(s*640,66))
  draw.text((15,5),'BOX SUPPORT REFERENCE | two front wheel surfaces support the box bottom',fill='black',font=font)
  draw.text((15,32),f'Kinematic preview, 2x speed | t={z["time"][k]:.2f}s | 24 x 44 x 18 cm box | dynamics not implied',fill='#666666',font=font)
  writer.append_data(np.array(out))
  if k%32==0:frames.append(np.array(out.resize((960,410))))
  if k in [0,624,1440,2320]:out.save(P/f'box_frame_{k:04d}.png')
io.mimsave(P/'box_preview.gif',frames,duration=133,loop=0);r.close()
