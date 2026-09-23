from pathlib import Path
import numpy as np,json
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;z=np.load(P/'reference.npz');h=np.load(P.parent/'human_biwheel/human_source.npz');t=np.arange(len(h['joints_w']))/30;s=z['source_time'];ground=np.median(h['object_pos'][:25,2]);raw=np.maximum(h['object_pos'][:,2]-ground,0);height=(z['object_pos'][:,2]-.13)/.43
fig,ax=plt.subplots(3,1,figsize=(10,8),sharex=True)
ax[0].plot(t,raw,label='Human object lift above rest',color='#bb7722');ax[0].plot(s,height,'--',label='Robot lift / 0.43',color='#257d99');ax[0].set_ylabel('Lift, source-scale m');ax[0].legend()
ax[1].plot(t,np.degrees(z['human_lean']),label='Human torso lean',color='#bb7722');rot=R.from_quat(z['qpos'][:,[4,5,6,3]]);up=rot.apply(np.tile([1,0,0.],(len(s),1)));lean=np.arctan2(np.linalg.norm(up[:,:2],axis=1),up[:,2]);ax[1].plot(s,np.degrees(lean),label='Robot lean = clip(0.65 x human)',color='#257d99');ax[1].set_ylabel('Forward lean, degrees');ax[1].legend()
ax[2].plot(s,z['object_pos'][:,2]-.13,label='Robot box bottom');ax[2].axhline(0,color='black',lw=.8);ax[2].set_ylabel('Box bottom, m');ax[2].set_xlabel('Human source time, seconds (robot takes 4x longer)');ax[2].legend()
for a in ax:
 a.axvspan(34/30,153/30,color='#aaaaaa',alpha=.12);a.grid(alpha=.2)
fig.suptitle('Same source timing; ground endpoints; continuous human lift curve');fig.tight_layout();fig.savefig(P/'alignment.png',dpi=160)
raw_interp=np.interp(s,t,raw);difference=height-raw_interp
r=json.loads((P/'report.json').read_text());r.update(normalized_lift_curve_rms_error_source_m=float(np.sqrt(np.mean(difference**2))),ground_endpoints_exact=bool(np.max(np.abs(z['object_pos'][[0,-1],2]-.13))<1e-9),human_object_rotation_preserved=False,horizontal_object_path='existing rolling-constrained axle path + scaled human object forward reach, no extra detour',grasp_location='upper rear corners, wheel inner faces; 9 cm behind box centre',scope_limit='kinematic reference passed; full free-payload controller did not pass')
(P/'report.json').write_text(json.dumps(r,indent=2))
