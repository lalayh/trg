import collections
import time
from datetime import datetime
import uuid

import numpy as np
import pandas as pd
import tqdm

from trg import io, vis
from trg.grasp import *
from trg.simulation import ClutterRemovalSim
from trg.utils.transform import Rotation, Transform
from trg.utils import visual
import matplotlib.pyplot as plt
from trg.utils.implicit import get_mesh_pose_list_from_world, get_scene_from_mesh_pose_list

MAX_CONSECUTIVE_FAILURES = 2

State = collections.namedtuple("State", ["tsdf", "pc"])


def run(
    grasp_plan_fn,
    logdir,
    description,
    scene,
    object_set,
    num_objects=5,
    n=6,
    N=None,
    num_rounds=1000,
    seed=1,
    sim_gui=False,
    rviz=False,
    sideview=False,
    visualize=False
):
    """Run several rounds of simulated clutter removal experiments.

    Each round, m objects are randomly placed in a tray. Then, the grasping pipeline is
    run until (a) no objects remain, (b) the planner failed to find a grasp hypothesis,
    or (c) maximum number of consecutive failed grasp attempts.
    """
    sim = ClutterRemovalSim(scene, object_set, gui=sim_gui, seed=seed, sideview=sideview)
    logger = Logger(logdir, description)
    cnt = 0
    success = 0
    left_objs = 0
    total_objs = 0
    cons_fail = 0
    no_grasp = 0
    planning_times = []
    total_times = []
    score_all = []

    for _ in tqdm.tqdm(range(num_rounds)):
        sim.reset(num_objects)

        round_id = logger.last_round_id() + 1
        logger.log_round(round_id, sim.num_objects)
        total_objs += sim.num_objects
        consecutive_failures = 1
        last_label = None
        trial_id = -1

        while sim.num_objects > 0 and consecutive_failures < MAX_CONSECUTIVE_FAILURES:

            trial_id += 1
            timings = {}

            # scan the scene
            tsdf, pc, timings["integration"] = sim.acquire_tsdf(n=n, N=N)
            state = State(tsdf, pc)

            if pc.is_empty():
                break  # empty point cloud, abort this round TODO this should not happen

            # visualize scene, plan grasps
            if rviz:
                vis.clear()
                vis.draw_workspace(sim.size)
                vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
                vis.draw_points(np.asarray(pc.points))

            if visualize:
                mesh_pose_list = get_mesh_pose_list_from_world(sim.world, object_set)
                scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list)
                grasps, scores, timings["planning"], visual_mesh = grasp_plan_fn(state, scene_mesh)
                scene_mesh.show()
                visual_mesh.show()
                logger.log_mesh(scene_mesh, visual_mesh, f'round_{round_id:03d}_trial_{trial_id:03d}')
            else:
                grasps, scores, timings["planning"] = grasp_plan_fn(state)

            planning_times.append(timings["planning"])
            total_times.append(timings["planning"] + timings["integration"])

            if len(grasps) == 0:
                no_grasp += 1
                break  # no detections found, abort this round

            if rviz:
                vis.draw_grasps(grasps, scores, sim.gripper.finger_depth)

            # if visualize:
            #     # fig = visual.plot_tsdf_with_grasps(tsdf.get_grid()[0], grasps)
            #     fig = visual.plot_voxel_grid(tsdf.get_grid()[0])
            #     plt.show(block=True)
            #     plt.close(fig)

            # execute grasp
            grasp, score = grasps[0], scores[0]

            if rviz:
                vis.draw_grasp(grasp, score, sim.gripper.finger_depth)

            label, _ = sim.execute_grasp(grasp, allow_contact=True)
            cnt += 1
            if label != Label.FAILURE:
                success += 1

            # log the grasp
            logger.log_grasp(round_id, state, timings, grasp, score, label)
            score_all.append(score)

            if last_label == Label.FAILURE and label == Label.FAILURE:
                consecutive_failures += 1
            else:
                consecutive_failures = 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                cons_fail += 1
            last_label = label
        left_objs += sim.num_objects
    success_rate = 100.0 * success / cnt
    declutter_rate = 100.0 * success / total_objs

    return success_rate, declutter_rate, np.mean(planning_times), np.mean(total_times), score_all


class Logger(object):
    def __init__(self, root, description):
        time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        description = "{}_{}".format(time_stamp, description).strip("_")

        self.logdir = root / description
        self.scenes_dir = self.logdir / "scenes"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.mesh_dir = self.logdir / "meshes"
        self.mesh_dir.mkdir(parents=True, exist_ok=True)

        self.rounds_csv_path = self.logdir / "rounds.csv"
        self.grasps_csv_path = self.logdir / "grasps.csv"
        self._create_csv_files_if_needed()

    def _create_csv_files_if_needed(self):
        if not self.rounds_csv_path.exists():
            io.create_csv(self.rounds_csv_path, ["round_id", "object_count"])

        if not self.grasps_csv_path.exists():
            columns = [
                "round_id",
                "scene_id",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "width",
                "score",
                "label",
                "integration_time",
                "planning_time",
            ]
            io.create_csv(self.grasps_csv_path, columns)

    def last_round_id(self):
        df = pd.read_csv(self.rounds_csv_path)
        return -1 if df.empty else df["round_id"].max()

    def log_round(self, round_id, object_count):
        io.append_csv(self.rounds_csv_path, round_id, object_count)

    def log_mesh(self, scene_mesh, aff_mesh, name):
        scene_mesh.export(self.mesh_dir / (name + "_scene.obj"))
        aff_mesh.export(str(self.mesh_dir / (name + "_aff.obj")))

    def log_grasp(self, round_id, state, timings, grasp, score, label):
        # log scene
        tsdf, points = state.tsdf, np.asarray(state.pc.points)
        scene_id = uuid.uuid4().hex
        scene_path = self.scenes_dir / (scene_id + ".npz")
        np.savez_compressed(scene_path, grid=tsdf.get_grid(), points=points)

        # log grasp
        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        x, y, z = grasp.pose.translation
        width = grasp.width
        label = int(label)
        io.append_csv(
            self.grasps_csv_path,
            round_id,
            scene_id,
            qx,
            qy,
            qz,
            qw,
            x,
            y,
            z,
            width,
            score,
            label,
            timings["integration"],
            timings["planning"],
        )


class Data(object):
    """Object for loading and analyzing experimental data."""

    def __init__(self, logdir):
        self.logdir = logdir
        self.rounds = pd.read_csv(logdir / "rounds.csv")
        self.grasps = pd.read_csv(logdir / "grasps.csv")

    def num_rounds(self):
        return len(self.rounds.index)

    def num_grasps(self):
        return len(self.grasps.index)

    def success_rate(self):
        return self.grasps["label"].mean() * 100

    def percent_cleared(self):
        df = (
            self.grasps[["round_id", "label"]]
            .groupby("round_id")
            .sum()
            .rename(columns={"label": "cleared_count"})
            .merge(self.rounds, on="round_id")
        )
        return df["cleared_count"].sum() / df["object_count"].sum() * 100

    def avg_planning_time(self):
        return self.grasps["planning_time"].mean()

    def read_grasp(self, i):
        scene_id, grasp, label = io.read_grasp(self.grasps, i)
        score = self.grasps.loc[i, "score"]
        scene_data = np.load(self.logdir / "scenes" / (scene_id + ".npz"))

        return scene_data["points"], grasp, score, label
