from pathlib import Path
import pandas as pd
import numpy as np
from trg.utils.transform import Rotation
import argparse
import os


def main(args):
    data_path = os.path.join(args.dataset, f"data_{args.scene}_train_processed_dex_noise")
    df = pd.read_csv(Path(data_path) / "grasps.csv")
    df_scene = df.groupby("scene_id").count()
    df_new = pd.DataFrame(data=None, index=None, columns=["scene_id"])
    for i in df_scene.index:
        item = pd.DataFrame(data=[i], columns=["scene_id"])
        df_new = pd.concat([df_new, item])
    df_new.to_csv(os.path.join(data_path, "grasp.csv"), index=False)

    os.makedirs(os.path.join(data_path, "masks"), exist_ok=True)
    for i, j in enumerate(df_scene.index):
        label = np.zeros([2, 40, 40, 40]).astype(np.longlong)
        rotations = np.zeros([2, 4, 40, 40, 40]).astype(np.single)
        width = np.zeros([40, 40, 40]).astype(np.single)
        label[1] = 1
        new_df = df.loc[df.loc[:, "scene_id"] == j, :]
        for k in range(len(new_df)):
            k_i = new_df.index[k]
            ori = Rotation.from_quat(new_df.loc[k_i, "qx":"qw"].to_numpy(np.single))
            pos = new_df.loc[k_i, "i":"k"].to_numpy(np.single)
            width_k = new_df.loc[k_i, "width"].astype(np.single)
            label_k = new_df.loc[k_i, "label"].astype(np.longlong)
            index = np.round(pos).astype(np.longlong)
            rotations_k = np.empty((2, 4), dtype=np.single)
            R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
            rotations_k[0] = ori.as_quat()
            rotations_k[1] = (ori * R).as_quat()
            label[0, index[0], index[1], index[2]] = label_k
            label[1, index[0], index[1], index[2]] = 0
            rotations[:, :, index[0], index[1], index[2]] = rotations_k
            width[index[0], index[1], index[2]] = width_k
        label_k = label[0]
        rotations_k = rotations.transpose((2, 3, 4, 0, 1))
        width_k = width
        index_k = label[1]
        np.savez(Path(data_path) / "masks" / j, label=label_k, rotations=rotations_k, width=width_k, index=index_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scene", type=str, default="packed")
    args = parser.parse_args()
    main(args)
