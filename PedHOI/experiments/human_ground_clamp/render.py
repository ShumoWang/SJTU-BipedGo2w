import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as io,trimesh
from scipy.spatial.transform import Rotation as R
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
P=Path(__file__).resolve().parent;z=np.load(P/'reference.npz');h=np.load(P.parent/'human_biwheel/human_source.npz');o=np.load(P/'source_object.npz');j=h['joints_w'];hull=trimesh.Trimesh(o['vertices_local'],o['faces'],process=False).convex_hull
m=mujoco.MjModel.from_xml_path(str(P/'scene.xml'));d=mujoco.MjData(m);m.vis.headlight.ambient[:]=.65;m.vis.global_.offwidth=640;m.vis.global_.offheight=480;r=mujoco.Renderer(m,480,640);cam=mujoco.MjvCamera();cam.distance=1.65;cam.elevation=-18
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);frames=[];parent=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19];fig=plt.figure(figsize=(6.4,4.8),dpi=100);ax=fig.add_subplot(111,projection='3d')
with io.get_writer(P/'human_ground_clamp.mp4',fps=15,macro_block_size=1) as writer:
 for f in range(0,len(j),2):
  k=f*16;q=z['qpos'][k];d.qpos[:23]=q;d.qpos[23:26]=z['object_pos'][k];d.qpos[26:]=z['object_quat'][k];mujoco.mj_forward(m,d)
  fw=R.from_quat(q[[4,5,6,3]]).apply([0,0,-1]);yaw=np.degrees(np.arctan2(fw[1],fw[0]));cam.azimuth=yaw+75;cam.lookat[:]=q[:3];cam.lookat[2]=.40;r.update_scene(d,camera=cam);robot=r.render().copy()
  ax.clear();center=j[f,0]*[1,1,0];jp=j[f]-center
  for i,pa in enumerate(parent):
   if pa>=0:ax.plot(*jp[[i,pa]].T,color='#4c5966',lw=3)
  vertices=hull.vertices@h['object_rot'][f].T+h['object_pos'][f]-center
  ax.add_collection3d(Poly3DCollection(vertices[hull.faces],facecolor='#b87534',edgecolor='#885728',linewidths=.15,alpha=.8));ax.scatter(*jp[[20,21]].T,c=['red','blue'],s=35)
  ax.set(xlim=(-.85,.85),ylim=(-.85,.85),zlim=(0,1.8));ax.set_box_aspect((1,1,1));ax.view_init(elev=12,azim=yaw+75);fig.tight_layout();fig.canvas.draw();human=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
  out=Image.new('RGB',(1280,546),'white');out.paste(Image.fromarray(human),(0,66));out.paste(Image.fromarray(robot),(640,66));draw=ImageDraw.Draw(out)
  phase='REACH / CLAMP ON GROUND' if f<34 else 'LIFT' if f<51 else 'CARRY' if f<=131 else 'LOWER TO GROUND' if f<156 else 'RELEASE / STAND'
  draw.text((15,5),'Source: raw OMOMO human + captured box',fill='black',font=font);draw.text((650,5),'Ground-to-ground side clamp (kinematic reference)',fill='black',font=font);draw.text((15,31),f'{phase} | source frame {f:03d} | source {f/30:.2f}s / robot {k/120:.2f}s | 4x robot preview',fill='#555555',font=font)
  writer.append_data(np.asarray(out));frames.append(np.asarray(out.resize((960,410))))
  if f in [0,26,32,52,90,148,154,160,192]:out.save(P/f'frame_{f:03d}.png')
io.mimsave(P/'preview.gif',frames,duration=67,loop=0);r.close();plt.close(fig)
