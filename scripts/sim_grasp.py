# import sys
# sys.path.remove("/home/yh/networks/giga/giga/src")
# sys.path.append("/home/yh/networks/trg/src")
import argparse
from pathlib import Path
from trg.detection import TRG
from trg.experiments import clutter_removal
import os
import numpy as np
import random
import torch
import json
import matplotlib.pyplot as plt


def main(args):

    grasp_planner = TRG(args.model, distant=args.dis, force_detection=args.force, visualize=args.vis, calibration=args.calibration, rviz=args.rviz)
    seed = args.seed
    gsr_all = []
    conf_all = []
    bm = []
    ege = 0
    ege_half = 0
    for y in range(int(1 / args.dis)):
    # for y in [4]:  # 4:giga,def select要修改一下;8.5,8,7:vgn3个阈值
        grasp_planner.qual_th = (y + 1) * args.dis
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        success_rate, declutter_rate, planning_time, total_time, score_all = clutter_removal.run(
            grasp_plan_fn=grasp_planner,
            logdir=args.logdir,
            description=args.description,
            scene=args.scene,
            object_set=args.object_set,
            num_objects=args.num_objects,
            n=args.num_view,
            num_rounds=args.num_rounds,
            seed=seed,
            sim_gui=args.sim_gui,
            rviz=args.rviz,
            sideview=args.sideview,
            visualize=args.vis
        )
        bm.append(len(score_all))
        gsr_all.append(success_rate)
        conf_all.append(np.mean(score_all) * 100)
        results = {
            'conf': np.mean(score_all) * 100,
            'gsr': success_rate,
            'dr': declutter_rate,
            'pt': planning_time * 1000,
            'tt': total_time * 1000
        }
        print('Average results:')
        print(f'Confidence: {np.mean(score_all) * 100:.2f} %')
        print(f'Grasp sucess rate: {success_rate:.2f} %')
        print(f'Declutter rate: {declutter_rate:.2f} %')
        print(f'Planning time: {planning_time * 1000:.2f}')
        print(f'Total time: {total_time * 1000:.2f}')
        with open("{}{}{}".format(args.result_path, y, ".json"), 'w') as f:
            json.dump(results, f, indent=2)
            f.close()
        with open(args.result_path, 'w') as g:
            json.dump(bm, g, indent=2)
            g.close()
    for o in range(int(1 / args.dis)):
        ege += bm[o] * abs(gsr_all[o] - conf_all[o]) / sum(bm)
    for p in range(int(1 / args.dis) - 5):
        ege_half += bm[5:][p] * abs(gsr_all[5:][p] - conf_all[5:][p]) / sum(bm[5:])

    conf_all = np.round(conf_all, 2)
    gsr_all = np.round(gsr_all, 2)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.bar(np.arange(int(1 / args.dis)) + 0.00, conf_all, color='orange', width=0.4, label="conf")
    plt.bar(np.arange(int(1 / args.dis)) + 0.40, gsr_all, color='royalblue', width=0.4, label="success rate")
    for a, b in zip(np.arange(int(1 / args.dis)) - 0.04, conf_all):
        plt.text(a, b, b, ha="center", va="bottom", fontsize=5)
    for j, k in zip(np.arange(int(1 / args.dis)) + 0.44, gsr_all):
        plt.text(j, k, k, ha="center", va="bottom", fontsize=5)

    plt.title('EGE  {:.2f},  EGE_half  {:.2f}'.format(ege, ege_half))
    plt.xticks(np.arange(int(1 / args.dis)) + 0.2, [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    plt.legend(loc="best")
    plt.savefig(f"./data/picture/{args.scene}.png", dpi=700)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, default="data/experiments")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    parser.add_argument("--object-set", type=str, default="blocks")
    parser.add_argument("--num-objects", type=int, default=5)
    parser.add_argument("--num-rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--rviz", action="store_true")
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--result-path", type=str)
    parser.add_argument("--num-view", type=int, default=1)
    parser.add_argument("--dis", type=float, default=0.1)
    parser.add_argument("--sideview",
                        action="store_true",
                        help="Whether to look from one side")
    parser.add_argument("--vis",
                        action="store_true",
                        help="visualize and save affordance")
    parser.add_argument(
        "--force",
        action="store_true",
        help=
        "When all grasps are under threshold, force the detector to select the best grasp"
    )
    args = parser.parse_args()
    main(args)
