import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import numpy as np,mujoco,json
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as io
P=Path(__file__).resolve().parent
m=mujoco.MjModel.from_xml_path('/home/robotennis2025/SJTU-BipedGo2w/GMR/assets/unitree_go2w/go2w.xml');d=mujoco.MjData(m)
m.vis.headlight.ambient[:]=.6;m.vis.headlight.diffuse[:]=.8;m.vis.global_.offwidth=640;m.vis.global_.offheight=480
r=mujoco.Renderer(m,height=480,width=640);cam=mujoco.MjvCamera();cam.distance=2.;cam.elevation=-15;cam.azimuth=25
z=np.load(P/'reference.npz');h=np.load(P/'human_source.npz');j=h['joints_w'];parent=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
fig=plt.figure(figsize=(6.4,4.8),dpi=100);ax=fig.add_subplot(111,projection='3d');frames=[]
for f in range(0,len(j),2):
 k=f*16;q=z['qpos'][k];d.qpos[:]=q;mujoco.mj_forward(m,d);cam.lookat[:]=q[:3];r.update_scene(d,camera=cam);robot=r.render().copy()
 ax.clear();jp=j[f]-j[f,0]*[1,1,0]
 for i,pa in enumerate(parent):
  if pa>=0:ax.plot(*jp[[i,pa]].T,color='#555555',lw=3)
 obj=h['object_pos'][f]-j[f,0]*[1,1,0];ax.scatter(*obj,color='purple',marker='s',s=80,label='Source object COM')
 ax.set(xlim=(-.9,.9),ylim=(-.9,.9),zlim=(0,1.8),xlabel='x',ylabel='y');ax.set_box_aspect((1,1,1));ax.view_init(elev=15,azim=25);ax.legend(loc='upper left',fontsize=9);fig.tight_layout();fig.canvas.draw();human=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
 out=Image.new('RGB',(1280,546),'white');out.paste(Image.fromarray(human),(0,65));out.paste(Image.fromarray(robot),(640,65));draw=ImageDraw.Draw(out)
 phase='REACH / PICK' if f<39 else 'LIFT' if f<51 else 'CARRY' if f<=131 else 'PLACE' if f<=145 else 'RELEASE / RETRACT'
 draw.text((15,4),'Raw OMOMO -> SMPL-X human',fill='black',font=font);draw.text((655,4),'Bi-wheel reference (KINEMATIC preview)',fill='black',font=font)
 draw.text((15,29),f'{phase} | source {f/30:.2f}s | robot {k/120:.2f}s | robot preview at 4x speed',fill='#555555',font=font)
 frames.append(np.asarray(out))
 if f in [26,86,144]:out.save(P/f'keyframe_{f:03d}.png')
io.mimsave(P/'human_to_biwheel.gif',frames,duration=67,loop=0)
io.mimsave(P/'human_to_biwheel.mp4',frames,fps=15,macro_block_size=1)
r.close();print('rendered human-to-biwheel preview')
