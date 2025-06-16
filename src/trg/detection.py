import time

import numpy as np
from scipy import ndimage
import torch
import matplotlib.pyplot as plt
from trg import vis
from trg.grasp import *
from trg.utils.transform import Transform, Rotation
from trg.networks import load_inference_network, load_calibration_inference_network
from trg.utils import visual
import trimesh


class TRG(object):
    def __init__(self, model_path, qual_th=0.9, distant=0.1, force_detection=False, visualize=False, calibration=False, rviz=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if calibration:
            self.net = load_calibration_inference_network(model_path, self.device)
        else:
            self.net = load_inference_network(model_path, self.device)
        self.net.eval()
        self.qual_th = qual_th
        self.distant = distant
        self.force_detection = force_detection
        self.rviz = rviz
        self.visualize = visualize

    def __call__(self, state, scene_mesh=None, aff_kwargs={}):
        tsdf_vol = state.tsdf.get_grid()
        voxel_size = state.tsdf.voxel_size
        size = state.tsdf.size

        tic = time.time()
        qual_vol, rot_vol, width_vol, regulatory_factors, before_calibration_vol = predict(tsdf_vol, self.net, self.device)
        qual_vol, rot_vol, width_vol, regulatory_factors, before_calibration_vol = process(tsdf_vol, qual_vol, rot_vol, width_vol, regulatory_factors, before_calibration_vol)

        if self.visualize:
            colored_scene_mesh = visual.affordance_visual(
                qual_vol, rot_vol.transpose(1, 2, 3, 0),
                scene_mesh, size, 40, **aff_kwargs)

        grasps, scores = select(qual_vol.copy(), rot_vol, width_vol, threshold=self.qual_th,
                                dis=self.distant, force_detection=self.force_detection)
        toc = time.time() - tic

        grasps, scores = np.asarray(grasps), np.asarray(scores)

        if len(grasps) > 0:
            p = np.random.permutation(len(grasps))  
            grasps = [from_voxel_coordinates(g, voxel_size) for g in grasps[p]]
            scores = scores[p]

        if self.rviz:
            vis.draw_quality(qual_vol, state.tsdf.voxel_size, threshold=0.01)

        if self.visualize:
            fig = visual.plot_tsdf_qual_vol(qual_vol)
            plt.show(block=True)
            plt.close(fig)
            fig = visual.plot_tsdf_qual_vol(before_calibration_vol)
            plt.show(block=True)
            plt.close(fig)
            fig = visual.plot_tsdf_regulatory_factors(regulatory_factors)
            plt.show(block=True)
            plt.close(fig)
            grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
            composed_scene = trimesh.Scene(colored_scene_mesh)
            for i, g_mesh in enumerate(grasp_mesh_list):
                composed_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
            return grasps, scores, toc, composed_scene
        else:
            return grasps, scores, toc


def predict(tsdf_vol, net, device):
    assert tsdf_vol.shape == (1, 40, 40, 40)

    # move input to the GPU
    tsdf_vol = torch.from_numpy(tsdf_vol).unsqueeze(0).to(device)  # （1，1，40，40，40）

    # forward pass
    with torch.no_grad():
        qual_vol, rot_vol, width_vol, regulatory_factors, before_calibration = net(tsdf_vol)
        qual_vol = torch.sigmoid(qual_vol)
        before_calibration = torch.sigmoid(before_calibration)

    # move output back to the CPU
    qual_vol = qual_vol.cpu().squeeze().numpy()
    rot_vol = rot_vol.cpu().squeeze().numpy()
    width_vol = width_vol.cpu().squeeze().numpy()
    regulatory_factors = regulatory_factors.cpu().squeeze().numpy()
    before_calibration_vol = before_calibration.cpu().squeeze().numpy()
    return qual_vol, rot_vol, width_vol, regulatory_factors, before_calibration_vol


def process(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    regulatory_factors,
    before_calibration_vol,
    gaussian_filter_sigma=1.0,
    min_width=1.33,
    max_width=9.33,
):
    tsdf_vol = tsdf_vol.squeeze()

    # smooth quality volume with a Gaussian
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )
    before_calibration_vol = ndimage.gaussian_filter(
        before_calibration_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > 0.5
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < 0.5)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0
    regulatory_factors[valid_voxels == False] = 0.0
    before_calibration_vol[valid_voxels == False] = 0.0

    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0
    regulatory_factors[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0
    before_calibration_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol, regulatory_factors, before_calibration_vol


# def select(qual_vol, rot_vol, width_vol, threshold=0.9, max_filter_size=4, dis=0.2, force_detection=False):
#     # threshold on grasp quality
#     qual_vol[qual_vol < threshold] = 0.0
#
#     max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
#     qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
#     mask = np.where(qual_vol, 1.0, 0.0)
#
#     grasps, scores = [], []
#     for index in np.argwhere(mask):
#         grasp, score = select_index(qual_vol, rot_vol, width_vol, index)
#         grasps.append(grasp)
#         scores.append(score)
#
#     sorted_grasps = [grasps[i] for i in reversed(np.argsort(scores))]
#     sorted_scores = [scores[i] for i in reversed(np.argsort(scores))]
#     if len(sorted_grasps) > 0:
#         sorted_grasps = [sorted_grasps[0]]
#         sorted_scores = [sorted_scores[0]]
#     return sorted_grasps, sorted_scores
#
#     # return grasps, scores


def select(qual_vol, rot_vol, width_vol, threshold=0.9, max_filter_size=4, dis=0.1, force_detection=False):
    qual_vol[qual_vol > threshold] = 0.0
    qual_vol[qual_vol <= threshold - dis] = 0.0

    # non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    # construct grasps
    grasps, scores = [], []
    for index in np.argwhere(mask):
        grasp, score = select_index(qual_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)

    sorted_grasps = [grasps[i] for i in reversed(np.argsort(scores))]
    sorted_scores = [scores[i] for i in reversed(np.argsort(scores))]

    if force_detection and len(sorted_grasps) > 0:
        sorted_grasps = [sorted_grasps[0]]
        sorted_scores = [sorted_scores[0]]

    return sorted_grasps, sorted_scores


def select_index(qual_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[:, i, j, k])
    pos = np.array([i, j, k], dtype=np.float64)
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score
