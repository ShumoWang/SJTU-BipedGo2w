import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import mujoco,numpy as np,imageio.v2 as io
from PIL import Image,ImageDraw,ImageFont
p=Path(__file__).resolve().parent
m=mujoco.MjModel.from_xml_path('/home/robotennis2025/SJTU-BipedGo2w/GMR/assets/unitree_go2w/go2w.xml');d=mujoco.MjData(m)
m.vis.headlight.ambient[:]=.6;m.vis.headlight.diffuse[:]=.8
m.vis.global_.offwidth=640;m.vis.global_.offheight=480
r=mujoco.Renderer(m,height=480,width=640)
a=np.load(p/'go2w_framewise.npz')['qpos'];b=np.load(p/'go2w_carry_adapted.npz')['qpos']
c=mujoco.MjvCamera();c.lookat[:]=[.35,-.35,.5];c.distance=2.5;c.azimuth=25;c.elevation=-15
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
frames=[]
for f in range(0,len(a),3):
 out=Image.new('RGB',(1280,535),'#fafafa');draw=ImageDraw.Draw(out)
 for j,q in enumerate([a[f],b[f]]):
  d.qpos[:]=q;mujoco.mj_forward(m,d);r.update_scene(d,camera=c)
  out.paste(Image.fromarray(r.render()),(640*j,55))
 draw.text((18,5),'Framewise IK (prototype)',fill='black',font=font)
 draw.text((658,5),'Adapted: fixed hold / rear-wheel transport',fill='black',font=font)
 draw.text((18,29),f'{f/30:.2f} s | kinematic playback, no physics simulation',fill='#666666',font=font)
 frames.append(np.asarray(out))
 if f==174:out.save(p/'mesh_comparison.png')
io.mimsave(p/'mesh_comparison.gif',frames,duration=100,loop=0)
r.close()
print('Saved mesh_comparison.gif')
