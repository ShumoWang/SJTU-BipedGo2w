"""Conservative mesh-pair proximity for near-degenerate MuJoCo queries."""
import numpy as np
import mujoco

def pair_distance(model,data,first,second):
 manifold=[float(c.dist) for c in data.contact if {int(c.geom1),int(c.geom2)}=={int(first),int(second)}]
 if manifold:return min(manifold)
 segment=np.zeros(6);distance=float(mujoco.mj_geomDistance(model,data,int(first),int(second),1.,segment));separation=float(np.linalg.norm(segment[3:]-segment[:3]))
 # A zero scalar with distinct witness points is not certified zero separation.
 return max(distance,separation) if distance>=0 else distance

def object_outward_normal(model,data,wheel,obj):
 normals=[np.array(c.frame[:3])*(1 if c.geom1==obj else -1) for c in data.contact if {int(c.geom1),int(c.geom2)}=={int(wheel),int(obj)}]
 if normals:return normals,True
 segment=np.zeros(6);mujoco.mj_geomDistance(model,data,int(wheel),int(obj),1.,segment);normal=segment[:3]-segment[3:];length=np.linalg.norm(normal)
 return ([normal/length] if length>1e-12 else []),False
