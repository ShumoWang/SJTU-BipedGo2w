"""Explicit alternative ideal-wheel model. Original meshes remain visual only.
Not a calibration to hardware; this isolates model/contact discretization issues.
"""
from pathlib import Path
import xml.etree.ElementTree as ET,json
P=Path(__file__).resolve().parent
src=Path('/home/robotennis2025/SJTU-BipedGo2w/GMR/assets/unitree_go2w/go2w.xml')
tree=ET.parse(src);root=tree.getroot();root.find('compiler').set('meshdir',str(src.parent/'assets'))
changes=[]
for b in root.iter('body'):
 if b.get('name','').endswith('wheel_link'):
  for g in b.findall('geom'):g.set('contype','0');g.set('conaffinity','0')
  ET.SubElement(b,'geom',dict(name=b.get('name')+'_ideal_contact',type='cylinder',size='0.086 0.0259',quat='0.70710678 0.70710678 0 0',rgba='0 0 0 0',contype='1',conaffinity='1',condim='3',friction='0.8 0.005 0.0001',priority='2',solref='0.01 1',solimp='0.95 0.99 0.001'))
  changes.append(b.get('name'))
tree.write(P/'ideal_wheels.xml')
(P/'contact_model.json').write_text(json.dumps(dict(source=str(src),alternate=str(P/'ideal_wheels.xml'),changed_bodies=changes,reason='original wheel has two overlapping active mesh geoms; alternative uses one analytic cylinder, condim 3; visual meshes retained',not_hardware_calibrated=True,mass_inertia_actuation_unchanged=True),indent=2))
