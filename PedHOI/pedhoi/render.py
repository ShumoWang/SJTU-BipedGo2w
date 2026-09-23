"""Source / previous yaw-only object / corrected object at aligned source time."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import json
import numpy as np,mujoco,imageio.v2 as io
from scipy.spatial.transform import Rotation as R
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .object_prior import corners

def run(cfg,out,human,seed):
 out=Path(out);z=np.load(out/'reference.npz');old=np.load(seed);h=np.load(human);interaction=np.load(out/'interaction.npz');j=h['joints_w'];parent=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
 model=mujoco.MjModel.from_xml_path(str(out/'scene.xml'));d=mujoco.MjData(model);model.vis.global_.offwidth=640;model.vis.global_.offheight=480;model.vis.headlight.ambient[:]=.65;r=mujoco.Renderer(model,480,640);cam=mujoco.MjvCamera();cam.distance=1.75;cam.elevation=-18
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);fig=plt.figure(figsize=(6.4,4.8),dpi=100);ax=fig.add_subplot(111,projection='3d');verts=corners(interaction['source_box_half_size']);faces=[[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]];frames=[]
 with io.get_writer(out/'comparison.mp4',fps=15,macro_block_size=1) as writer:
  for f in range(0,len(j),2):
   k=round(f/cfg['source_fps']*cfg['time_scale']*cfg['output_fps']);q=z['qpos'][k];fw=R.from_quat(q[[4,5,6,3]]).apply([0,0,-1]);yaw=np.degrees(np.arctan2(fw[1],fw[0]));cam.azimuth=yaw+70;cam.lookat[:]=q[:3];cam.lookat[2]=.4
   outimg=Image.new('RGB',(1920,546),'white');draw=ImageDraw.Draw(outimg)
   for col,data in [(1,old),(2,z)]:
    d.qpos[:23]=data['qpos'][k];d.qpos[23:26]=data['object_pos'][k];d.qpos[26:]=data['object_quat'][k];mujoco.mj_forward(model,d);r.update_scene(d,camera=cam);outimg.paste(Image.fromarray(r.render()),(col*640,66))
   ax.clear();center=j[f,0]*[1,1,0];jp=j[f]-center
   for i,pa in enumerate(parent):
    if pa>=0:ax.plot(*jp[[i,pa]].T,color='#4c5966',lw=3)
   rot=R.from_quat(interaction['source_object_quat'][f,[1,2,3,0]]);v=rot.apply(verts)+h['object_pos'][f]-center;ax.add_collection3d(Poly3DCollection([v[face] for face in faces],facecolor='#b87534',edgecolor='#79552e',linewidths=.6,alpha=.8));ax.scatter(*jp[[20,21]].T,c=['red','blue'],s=30)
   ax.set(xlim=(-.85,.85),ylim=(-.85,.85),zlim=(0,1.8));ax.set_box_aspect((1,1,1));ax.view_init(elev=12,azim=yaw+70);fig.tight_layout();fig.canvas.draw();outimg.paste(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3]),(0,66))
   for x,title in [(15,'Raw human + canonical source object'),(650,'Previous: yaw-only object'),(1290,'Interaction + constrained 3D object pose')]:draw.text((x,5),title,fill='black',font=font)
   draw.text((15,32),f'Source frame {f:03d} | robot time {k/cfg["output_fps"]:.2f}s | KINEMATIC comparison, 4x robot speed | dynamics not validated',fill='#666666',font=font)
   writer.append_data(np.asarray(outimg));frames.append(np.asarray(outimg.resize((1200,342))))
   if f in [30,90,152,192]:outimg.save(out/f'comparison_{f:03d}.png')
 io.mimsave(out/'preview.gif',frames,duration=67,loop=0);r.close();plt.close(fig)
 # Scientific audit plot: prior error, ground clearance, bilateral gaps.
 val=np.load(out/'validation.npz');opt=np.load(out/'optimization.npz');s=z['source_time'];fig,ax=plt.subplots(3,1,figsize=(10,8),sharex=True)
 ax[0].plot(s,opt['orientation_error_before_deg'],label='Previous yaw-only');ax[0].plot(s,opt['orientation_error_after_deg'],label='Corrected');ax[0].set_ylabel('Object SO(3) error, deg');ax[0].legend()
 ax[1].plot(s,val['box_bottom'],label='Actual rotated box bottom');ax[1].plot(np.arange(len(j))/cfg['source_fps'],interaction['source_clearance']*cfg['clearance_scale'],'--',label='Scaled human clearance');ax[1].set_ylabel('Ground clearance, m');ax[1].legend()
 ax[2].plot(s,val['wheel_box_distance']*1000);ax[2].set_ylim(-2,5);ax[2].set_ylabel('Wheel / box gap, mm');ax[2].set_xlabel('Human source time, s')
 for a in ax:a.axvspan(34/30,153/30,color='gray',alpha=.12);a.grid(alpha=.2)
 fig.tight_layout();fig.savefig(out/'validation_summary.png',dpi=160);plt.close(fig)
if __name__=='__main__':
 import sys
 c=json.loads(Path(sys.argv[1]).read_text());run(c,c['output'],'experiments/human_biwheel/human_source.npz','experiments/human_ground_clamp/reference.npz')
