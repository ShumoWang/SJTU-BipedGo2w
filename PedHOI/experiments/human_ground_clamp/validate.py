from pathlib import Path
import json,numpy as np,mujoco
P=Path(__file__).resolve().parent;z=np.load(P/'reference.npz');m=mujoco.MjModel.from_xml_path(str(P/'scene.xml'));d=mujoco.MjData(m);r=json.loads((P/'report.json').read_text());obj=m.geom('payload_box').id;gids=[np.where(m.geom_bodyid==m.body(leg+'_wheel_link').id)[0][0] for leg in ['FL','FR']];dist=[];boxpen=0.;robotpen=0.;floorpen=0.;bad=[]
for k,q in enumerate(z['qpos']):
 d.qpos[:23]=q;d.qpos[23:26]=z['object_pos'][k];d.qpos[26:]=z['object_quat'][k];mujoco.mj_forward(m,d);
 distances=[]
 for g in gids:
  manifold=[c.dist for c in d.contact if set([int(c.geom1),int(c.geom2)])==set([int(g),obj])]
  distances.append(min(manifold) if manifold else mujoco.mj_geomDistance(m,d,int(g),obj,1,None))
 dist.append(distances)
 for c in d.contact:
  a,b=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2];penetration=max(0.,-c.dist)
  if obj in [c.geom1,c.geom2]:boxpen=max(boxpen,penetration)
  elif a and b:robotpen=max(robotpen,penetration)
  else:floorpen=max(floorpen,penetration)
  if c.dist<-.002:bad.append((float(c.dist),k,m.body(a).name,m.body(b).name))
ids=np.array([i for i in range(16) if i%4!=3]);lim=m.jnt_range[1:17];jerr=max(np.max(lim[ids,0]-z['qpos'][:,7+ids]),np.max(z['qpos'][:,7+ids]-lim[ids,1]),0);dist=np.array(dist);contact=(z['source_time']>=34/30)&(z['source_time']<=153/30)
r.update(box_penetration_max_m=boxpen,robot_self_penetration_max_m=robotpen,robot_floor_penetration_max_m=floorpen,joint_limit_violation_max_rad=float(jerr),closed_wheel_box_distance_max_m=float(np.abs(dist[contact]).max()),worst_collisions=sorted(bad)[:12]);r['geometry_validated']=bool(boxpen<.002 and robotpen<.002 and floorpen<.002 and jerr<1e-5 and r['closed_wheel_box_distance_max_m']<.003 and r['rear_no_slip_max_m_s']<.003)
np.savez(P/'geometry_check.npz',wheel_box_distance=dist);(P/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))

assert r["geometry_validated"], "Ground-clamp geometry validation failed"
assert np.max(np.abs(z["object_pos"][[0,-1],2]-z["object_half_size"][2]))<1e-9, "Box endpoints are not on the ground"
