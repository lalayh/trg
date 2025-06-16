#!/usr/bin/env python

"""
Open-loop grasp execution using a Panda arm and wrist-mounted RealSense camera.
"""

import argparse
from pathlib import Path
import time
import cv_bridge
import geometry_msgs.msg
import numpy as np
import rospy
import sensor_msgs.msg

from trg import vis
from trg.experiments.clutter_removal import State
from trg.detection import TRG
from trg.perception import *
from trg.utils import ros_utils
from trg.utils.transform import Rotation, Transform
from trg.utils.panda_control import PandaCommander

# tag lies on the table in the center of the workspace
T_base_tag = Transform(Rotation.identity(), [0.42, 0.0, 0.195])
round_id = 0


class PandaGraspController(object):
    def __init__(self, args):

        self.base_frame_id = rospy.get_param("~base_frame_id")
        self.tool0_frame_id = rospy.get_param("~tool0_frame_id")
        self.T_tool0_tcp = Transform.from_dict(
            rospy.get_param("~T_tool0_tcp"))  # TODO
        self.T_tcp_tool0 = self.T_tool0_tcp.inverse()
        self.finger_depth = rospy.get_param("~finger_depth")
        self.size = 6.0 * self.finger_depth
        self.scan_joints = rospy.get_param("~scan_joints")

        self.setup_panda_control()
        self.tf_tree = ros_utils.TransformTree()
        self.define_workspace()
        self.create_planning_scene()
        self.tsdf_server = TSDFServer()
        self.plan_grasps = TRG(args.model, qual_th=args.interval_upper, distant=args.dis, calibration=args.calibration, rviz=True)

        rospy.loginfo("Ready to take action")

        self.two_failures = 0
        self.no_grasps = 0
        self.last_label = None
        self.conf_gsr = []

    def setup_panda_control(self):
        rospy.Subscriber(
            "/joint_states", sensor_msgs.msg.JointState, self.joints_cb, queue_size=1
        )
        self.pc = PandaCommander()
        self.pc.move_group.set_end_effector_link(self.tool0_frame_id)

    def define_workspace(self):
        z_offset = -0.06
        t_tag_task = np.r_[[-0.5 * self.size, -0.5 * self.size, z_offset]]
        T_tag_task = Transform(Rotation.identity(), t_tag_task)
        self.T_base_task = T_base_tag * T_tag_task

        self.tf_tree.broadcast_static(self.T_base_task, self.base_frame_id, "task")

        rospy.sleep(1.0)  # wait for the TF to be broadcasted

    def create_planning_scene(self):
        # collision box for table
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = self.base_frame_id
        msg.pose = ros_utils.to_pose_msg(T_base_tag)
        msg.pose.position.z -= 0.01
        self.pc.scene.add_box("table", msg, size=(0.6, 0.6, 0.02))

        rospy.sleep(1.0)  # wait for the scene to be updated

    def joints_cb(self, msg):
        self.gripper_width = msg.position[7] + msg.position[8]

    def run(self):
        vis.clear()
        vis.draw_workspace(self.size)
        self.pc.move_gripper(0.08)
        self.pc.home()

        tsdf, pc = self.acquire_tsdf()
        vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
        vis.draw_points(np.asarray(pc.points))
        rospy.loginfo("Reconstructed scene")

        state = State(tsdf, pc)
        grasps, scores, planning_time = self.plan_grasps(state)

        vis.draw_grasps(grasps, scores, self.finger_depth)
        rospy.loginfo("Planned grasps")

        if len(grasps) == 0:
            self.no_grasps = 1
            rospy.loginfo("No grasps detected")
            return

        grasp, score = self.select_grasp(grasps, scores)
        vis.draw_grasp(grasp, score, self.finger_depth)
        rospy.loginfo("Selected grasp")

        self.pc.home()
        label = self.execute_grasp(grasp)
        rospy.loginfo("Grasp execution")

        if self.last_label is False and label is False:
            self.two_failures += 1
        if label:
            self.drop()
        self.last_label = label
        self.conf_gsr.append([score, int(label)])

        self.pc.home()

    def acquire_tsdf(self):
        self.pc.goto_joints(self.scan_joints[0])

        self.tsdf_server.reset()
        self.tsdf_server.integrate = True
        time.sleep(1)

        self.tsdf_server.integrate = False
        tsdf = self.tsdf_server.low_res_tsdf
        pc = self.tsdf_server.high_res_tsdf.get_cloud()

        return tsdf, pc

    def select_grasp(self, grasps, scores):
        # select the highest grasp
        heights = np.empty(len(grasps))
        for i, grasp in enumerate(grasps):
            heights[i] = grasp.pose.translation[2]
        idx = np.argmax(heights)
        grasp, score = grasps[idx], scores[idx]

        # make sure camera is pointing forward
        rot = grasp.pose.rotation
        axis = rot.as_matrix()[:, 0]
        if axis[0] < 0:
            grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)

        return grasp, score

    def execute_grasp(self, grasp):
        T_task_grasp = grasp.pose
        T_base_grasp = self.T_base_task * T_task_grasp

        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_base_pregrasp = T_base_grasp * T_grasp_pregrasp
        T_base_retreat = T_base_grasp * T_grasp_retreat

        self.pc.goto_pose(T_base_pregrasp * self.T_tcp_tool0, velocity_scaling=0.5)
        self.approach_grasp(T_base_grasp)

        self.pc.grasp(width=0.0, force=20.0)

        self.pc.goto_pose(T_base_retreat * self.T_tcp_tool0, acceleration_scaling=0.3)

        # lift hand
        T_retreat_lift_base = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
        T_base_lift = T_retreat_lift_base * T_base_retreat
        self.pc.goto_pose(T_base_lift * self.T_tcp_tool0)

        if self.gripper_width > 0.0005:
            return True
        else:
            return False

    def approach_grasp(self, T_base_grasp):
        self.pc.goto_pose(T_base_grasp * self.T_tcp_tool0, velocity_scaling=0.2, acceleration_scaling=0.2)

    def drop(self):
        self.pc.goto_joints(
            [-0.289, -0.128, -1.251, -2.061, -0.0952, 2.048, 0.527], 0.3, 0.3)

        self.pc.move_gripper(0.08)


class TSDFServer(object):
    def __init__(self):
        self.cam_frame_id = rospy.get_param("~cam/frame_id")
        self.cam_topic_name = rospy.get_param("~cam/topic_name")
        self.intrinsic = CameraIntrinsic.from_dict(rospy.get_param("~cam/intrinsic"))
        self.size = 6.0 * rospy.get_param("~finger_depth")

        self.cv_bridge = cv_bridge.CvBridge()
        self.tf_tree = ros_utils.TransformTree()
        self.integrate = False
        rospy.Subscriber(self.cam_topic_name, sensor_msgs.msg.Image, self.sensor_cb)

    def reset(self):
        self.low_res_tsdf = TSDFVolume(self.size, 40)
        self.high_res_tsdf = TSDFVolume(self.size, 120)

    def sensor_cb(self, msg):
        if not self.integrate:
            return

        img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001
        T_cam_task = self.tf_tree.lookup(
            self.cam_frame_id, "task", msg.header.stamp, rospy.Duration(0.1)
        )

        self.low_res_tsdf.integrate(img, self.intrinsic, T_cam_task)
        self.high_res_tsdf.integrate(img, self.intrinsic, T_cam_task)


def main(args):
    rospy.init_node("panda_grasp")
    panda_grasp = PandaGraspController(args)
    if not args.save_path.exists():
        np.save(args.save_path, np.array([[0, 0]]))
    result = np.load(args.save_path)

    while not rospy.is_shutdown():
        if panda_grasp.two_failures == 1 or panda_grasp.no_grasps == 1:
            break
        panda_grasp.run()
    result = np.append(result, np.array(panda_grasp.conf_gsr), axis=0)
    np.save(args.save_path, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--dis", type=float, default=0.1)
    parser.add_argument("--interval-upper", type=float, choices=[0.6, 0.7, 0.8, 0.9, 1.0], default=0.6)
    parser.add_argument("--calibration", action="store_true")
    args = parser.parse_args()
    main(args)
