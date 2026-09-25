"""Compare actual Isaac link poses with MuJoCo FK; no Isaac import needed.

Run after replay_npz_go2w.py: python tests/check_isaac_replay.py --help
"""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


def check(source, readback, model):
    with np.load(source, allow_pickle=False) as data:
        z = {k: data[k] for k in data.files}
    with np.load(readback, allow_pickle=False) as data:
        a = {k: data[k] for k in data.files}
    m = mujoco.MjModel.from_xml_path(str(model.resolve()))
    d = mujoco.MjData(m)
    errors = []
    for i, k in enumerate(a['indices']):
        d.qpos[:] = z['qpos'][k]
        mujoco.mj_forward(m, d)
        errors.append([np.linalg.norm(a['body_pos'][i, j] - d.xpos[m.body(str(name)).id])
                       for j, name in enumerate(a['body_names'])])
    ids = a['indices']
    def rotation_error(actual, expected):
        r = Rotation.from_quat(actual[:, [1, 2, 3, 0]])
        target = Rotation.from_quat(expected[:, [1, 2, 3, 0]])
        return float(np.rad2deg((r * target.inv()).magnitude()).max())
    report = dict(frames=len(ids), first_frame=int(ids[0]), last_frame=int(ids[-1]),
        body_position_max_m=float(np.max(errors)),
        object_position_max_m=float(np.abs(a['object_pose'][:, :3] - z['object_pos'][ids]).max()),
        root_rotation_max_deg=rotation_error(a['root_quat'], z['qpos'][ids, 3:7]),
        object_rotation_max_deg=rotation_error(a['object_pose'][:, 3:7], z['object_quat'][ids]),
        joint_max_rad=float(np.abs(a['joint_pos']-z['qpos'][ids, 7:]).max()))
    assert report['body_position_max_m'] < 1e-5, report
    assert report['object_position_max_m'] < 1e-5, report
    assert max(report['root_rotation_max_deg'], report['object_rotation_max_deg']) < 1e-3, report
    assert report['joint_max_rad'] < 1e-5, report
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--readback', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = check(args.source, args.readback, args.model)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
