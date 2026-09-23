"""Read-only no-slip kinematic screen for the generated references.
Uses actual MuJoCo world joint axes and point Jacobians, not torso heading.
Ideal thin circular wheel contacts on z=0; not a dynamic feasibility certificate.
"""
from pathlib import Path
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
MODEL='/home/robotennis2025/SJTU-BipedGo2w/GMR/assets/unitree_go2w/go2w.xml'
m=mujoco.MjModel.from_xml_path(MODEL);d=mujoco.MjData(m)
normal=np.array([0.,0.,1.]);radius=.086
reports={}
for folder,lo,hi in [('resmimic_carry',135,275),('resmimic_pick_place',100,141)]:
 z=np.load(BASE/folder/'go2w_carry_adapted.npz');q=z['qpos'];fps=float(z['fps']);dt=1/fps
 vel=np.zeros((len(q),m.nv))
 for f in range(len(q)):
  a=max(0,f-1);b=min(len(q)-1,f+1)
  mujoco.mj_differentiatePos(m,vel[f],(b-a)*dt,q[a],q[b])
 rows=[];positions=[];axes=[];directions=[];hubvel=[];contacts=[]
 for f in range(len(q)):
  d.qpos[:]=q[f];d.qvel[:]=vel[f];mujoco.mj_forward(m,d)
  rr=[];ps=[];axs=[];ts=[];vs=[];cs=[]
  heading=R.from_quat(q[f,[4,5,6,3]]).as_matrix() @ np.array([0.,0.,-1.])
  heading[2]=0;heading/=np.linalg.norm(heading)
  left=np.cross(normal,heading)
  for name in ['RL','RR']:
   jid=m.joint(name+'_wheel_joint').id;bid=m.body(name+'_wheel_link').id
   axle=d.xaxis[jid].copy();center=d.xanchor[jid].copy()
   lateral=axle-axle.dot(normal)*normal;lateral/=np.linalg.norm(lateral)
   rolling=np.cross(lateral,normal)
   radial=normal-axle.dot(normal)*axle;radial=-radius*radial/np.linalg.norm(radial)
   cp=center+radial
   jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
   mujoco.mj_jac(m,d,jp,jr,center,bid);vc=jp@vel[f]
   mujoco.mj_jac(m,d,jp,jr,cp,bid);vcontact=jp@vel[f]
   # Best spin-only correction with all other generalized velocities fixed.
   column=jp[:,m.jnt_dofadr[jid]];correction=-column[:2].dot(vcontact[:2])/column[:2].dot(column[:2])
   best=vcontact+column*correction
   angle=np.degrees(np.arctan2(rolling.dot(left),rolling.dot(heading)))
   rr.append([vc.dot(lateral),vcontact.dot(rolling),np.linalg.norm(vcontact[:2]),angle,cp[2],np.linalg.norm(vc[:2]),np.linalg.norm(best[:2]),vel[f,m.jnt_dofadr[jid]],correction])
   ps.append(center);axs.append(lateral);ts.append(rolling);vs.append(vc);cs.append(cp)
  rows.append(rr);positions.append(ps);axes.append(axs);directions.append(ts);hubvel.append(vs);contacts.append(cs)
 rows=np.array(rows);positions=np.array(positions);axes=np.array(axes);directions=np.array(directions);hubvel=np.array(hubvel)
 seg=rows[lo:hi];moving=seg[:,:,5]>.05
 def stats(x):return dict(rms=float(np.sqrt(np.mean(x*x))),p95_abs=float(np.percentile(np.abs(x),95)),max_abs=float(np.max(np.abs(x))))
 reports[folder]=dict(frame_interval_zero_based=[lo,hi-1],time_interval_s=[lo/fps,(hi-1)/fps],lateral_slip_m_s=stats(seg[:,:,0]),longitudinal_contact_slip_m_s=stats(seg[:,:,1]),planar_contact_slip_m_s=stats(seg[:,:,2]),best_possible_slip_after_spin_only_correction_m_s=stats(seg[:,:,6]),toe_angle_deg_by_wheel=[[float(seg[:,j,3].min()),float(seg[:,j,3].max())] for j in range(2)],moving_wheel_samples=int(moving.sum()),fraction_moving_wheel_samples_above_2cm_s_lateral=float(np.mean(np.abs(seg[:,:,0][moving])>.02)),integrated_absolute_lateral_slip_m_per_wheel=(np.sum(np.abs(seg[:,:,0]),axis=0)*dt).tolist(),ideal_contact_z_range_m=[float(seg[:,:,4].min()),float(seg[:,:,4].max())],status='FAIL no-slip kinematic screen',threshold_note='2 cm/s is an illustrative engineering screen, not a universal physical limit')
 np.savez(OUT/(folder+'_diagnostics.npz'),metrics=rows,metric_names=np.array(['lateral_hub_speed','longitudinal_contact_slip','planar_contact_slip','toe_angle_deg','ideal_contact_z','hub_planar_speed','best_spin_only_contact_slip','wheel_joint_rate','best_spin_delta']),fps=fps,hub_position=positions,lateral_axis=axes,rolling_direction=directions,hub_velocity=hubvel)
 fig,ax=plt.subplots(3,1,figsize=(10,8),sharex=True)
 time=np.arange(len(q))/fps
 for j,name in enumerate(['rear left','rear right']):
  ax[0].plot(time,rows[:,j,3],label=name)
  ax[1].plot(time,rows[:,j,0],label=name)
  ax[2].plot(time,rows[:,j,1],label=name)
 for a in ax:a.axvspan(lo/fps,(hi-1)/fps,color='teal',alpha=.12);a.grid(alpha=.3);a.legend()
 ax[0].set_ylabel('Toe angle (deg)');ax[1].set_ylabel('Lateral slip (m/s)');ax[2].set_ylabel('Rolling residual (m/s)');ax[2].set_xlabel('Time (s)')
 fig.suptitle(folder+' | actual wheel axes + Jacobian contact velocity\nShaded: fixed-leg transport. Ideal wheel / flat-ground screen.')
 fig.tight_layout();fig.savefig(OUT/(folder+'_slip.png'),dpi=140);plt.close(fig)
 # Top-down worst frame: velocity vs allowed rolling directions.
 f=lo+np.argmax(np.max(np.abs(seg[:,:,0]),axis=1));fig,ax=plt.subplots(figsize=(7,6))
 origin=positions[f].mean(axis=0)
 for j,name in enumerate(['RL','RR']):
  p=positions[f,j]-origin;t=directions[f,j];v=hubvel[f,j];lat=axes[f,j]
  ax.plot([p[0]-.16*t[0],p[0]+.16*t[0]],[p[1]-.16*t[1],p[1]+.16*t[1]],color='teal',lw=3,label='Allowed rolling direction' if j==0 else None)
  ax.quiver(p[0],p[1],v[0]*.4,v[1]*.4,angles='xy',scale_units='xy',scale=1,color='darkorange',label='Actual hub velocity (0.4 s scale)' if j==0 else None)
  lateral=v.dot(lat)*lat
  ax.quiver(p[0],p[1],lateral[0]*.4,lateral[1]*.4,angles='xy',scale_units='xy',scale=1,color='crimson',label='Forbidden lateral component' if j==0 else None)
  ax.text(p[0],p[1]-.03,name)
 ax.set(xlim=(-.5,.5),ylim=(-.5,.5),xlabel='World X relative to rear midpoint (m)',ylabel='World Y (m)',title=f'{folder}: frame {f}, {f/fps:.2f} s\nWheel spin cannot cancel the red lateral component')
 ax.set_aspect('equal');ax.grid(alpha=.3);ax.legend(loc='upper left',fontsize=8);fig.tight_layout();fig.savefig(OUT/(folder+'_topview.png'),dpi=150);plt.close(fig)
(OUT/'report.json').write_text(json.dumps(dict(model=MODEL,mujoco_version=mujoco.__version__,method='mj_differentiatePos + actual world axle xaxis + mj_jac at ideal wheel-plane contact',wheel_radius_m=radius,not_dynamic_validation=True,results=reports),indent=2))
print(json.dumps(reports,indent=2))
