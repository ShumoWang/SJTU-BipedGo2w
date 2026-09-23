import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as io
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation as R
P=Path(__file__).resolve().parent;m=mujoco.MjModel.from_xml_path(str(P/'scene.xml'));d=mujoco.MjData(m);m.vis.global_.offwidth=800;m.vis.global_.offheight=600;m.vis.headlight.ambient[:]=.6;r=mujoco.Renderer(m,600,800);cam=mujoco.MjvCamera();cam.distance=1.65;cam.elevation=-20;rows=np.load(P/'free_box_rollout.npz')['records'];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
with io.get_writer(P/'free_box_test.mp4',fps=25,macro_block_size=1) as w:
 for row in rows[::4]:
  d.qpos[:]=row[1:31];d.qvel[:]=row[31:];mujoco.mj_forward(m,d);fw=R.from_quat(d.qpos[[4,5,6,3]]).apply([0,0,-1]);cam.azimuth=np.degrees(np.arctan2(fw[1],fw[0]))+95;cam.lookat[:]=d.qpos[:3]+[0,0,.08];r.update_scene(d,camera=cam);out=Image.new('RGB',(800,656),'white');out.paste(Image.fromarray(r.render()),(0,56));draw=ImageDraw.Draw(out);draw.text((10,5),'FREE 0.5 kg BOX | carry test | actual torque-driven simulation',fill='black',font=font);draw.text((10,29),f't={row[0]:.2f}s | box slips: NOT a successful manipulation rollout',fill='#a02222',font=font);w.append_data(np.asarray(out))
r.close()
