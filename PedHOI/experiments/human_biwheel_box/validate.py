from pathlib import Path
import json,numpy as np,mujoco
P=Path(__file__).resolve().parent;z=np.load(P/'reference.npz');m=mujoco.MjModel.from_xml_path(str(P/'scene.xml'));d=mujoco.MjData(m);r=json.loads((P/'report.json').read_text());penetration=0.;objpenetration=0.
for k,q in enumerate(z['qpos']):
 d.qpos[:23]=q;d.qpos[23:26]=z['object_pos'][k];d.qpos[26:]=z['object_quat'][k];mujoco.mj_forward(m,d)
 for c in d.contact:
  b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
  if b1 and b2:penetration=max(penetration,-c.dist)
  if m.geom('payload_box').id in [c.geom1,c.geom2]:objpenetration=max(objpenetration,-c.dist)
ids=[i for i in range(16) if i%4!=3];lim=m.jnt_range[1:17];assert np.all(z['qpos'][:,7+np.array(ids)]>=lim[ids,0]-1e-6);assert np.all(z['qpos'][:,7+np.array(ids)]<=lim[ids,1]+1e-6)
r.update(all_non_floor_penetration_max_m=float(penetration),all_box_contact_penetration_max_m=float(objpenetration));r['geometry_pass']=bool(penetration<.001 and objpenetration<.001 and r['front_surface_fit_max_m']<.001 and r['rear_no_slip_max_m_s']<.003)
(P/'report.json').write_text(json.dumps(r,indent=2));assert r['geometry_pass'];print('PASS:',r)
