import smplx
import numpy as np

model = smplx.create(
    model_path="/home/shumowang/Desktop/mimic_ws/GMR/assets/body_models",
    model_type="smplx",
    gender="neutral",
    ext="npz",
    use_pca=False,
    batch_size=1,
)

with np.printoptions(precision=6, suppress=True):
    output = model()
    joints = output.joints[0].detach().cpu().numpy()

    names = [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ]

    for i, name in enumerate(names):
        print(f"{i:2d} {name:16s}: {joints[i]}")