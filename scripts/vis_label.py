import numpy as np
import matplotlib.pylab as plt


label = np.load("./data/datasets/data_packed_train_processed_dex_noise/masks/0a0cccd3fb8445339db59e18403fc413.npz")["label"]
index = np.load("./data/datasets/data_packed_train_processed_dex_noise/masks/0a0cccd3fb8445339db59e18403fc413.npz")["index"]
positive_grasps = (label > 0)
positive_grasps_color = np.empty(positive_grasps.shape, dtype=object)
positive_grasps_color[positive_grasps] = 'blue'

all_grasps = (index == 0)
all_grasps_color = np.empty(all_grasps.shape, dtype=object)
all_grasps_color[all_grasps] = 'red'

space = (label >= 0)
space_color = np.empty(space.shape, dtype=object)
space_color[space] = 'whitesmoke'

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection='3d')
ax.voxels(space, facecolors=space_color, edgecolor='none', alpha=0.2, linewidth=0.1)
ax.voxels(all_grasps, facecolors=all_grasps_color, edgecolor='k', linewidth=0.2)
ax.voxels(positive_grasps, facecolors=positive_grasps_color, edgecolor='k', linewidth=0.2)
plt.axis("off")
plt.tight_layout()
plt.show()
