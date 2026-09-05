#!/usr/bin/env python3
"""Closed-loop RGB-D visual-servo grasp for the side-lying chips can.

The node repeatedly reads the current RGB-D image, estimates the chips can
position in base_link, and applies small Cartesian corrections to
gripper_center_link before closing the AG95 gripper.
"""

import copy
import math
import sys
import threading
from pathlib import Path

SYSTEM_DIST_PACKAGES = (
    "/usr/lib/python3/dist-packages",
    "/usr/local/lib/python3/dist-packages",
)


def _ensure_ros_python_paths():
    for path in SYSTEM_DIST_PACKAGES:
        if Path(path).is_dir() and path not in sys.path:
            sys.path.append(path)


_ensure_ros_python_paths()

import cv2
import message_filters
import moveit_commander
import moveit_msgs.msg
import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge
from dh_gripper_msgs.msg import GripperCtrl, GripperState
from sensor_msgs.msg import CameraInfo, Image


CONFIRM_TOKEN = "RUN_CHIPS_CAN_VISUAL_SERVO"
AG95_STROKE_M = 0.095


def width_to_driver_position(width_m):
    width_m = max(0.0, min(AG95_STROKE_M, float(width_m)))
    return int(round(width_m / AG95_STROKE_M * 1000.0))


def driver_position_to_width(position):
    position = max(0.0, min(1000.0, float(position)))
    return AG95_STROKE_M * position / 1000.0


def normalize(vector):
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError("Cannot normalize near-zero vector")
    return vector / norm


def quaternion_to_matrix(q):
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_from_matrix(rot):
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat


def rotation_from_x_down(jaw_axis):
    x_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    y_axis = np.asarray(jaw_axis, dtype=np.float64)
    y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
    y_axis = normalize(y_axis)
    z_axis = normalize(np.cross(x_axis, y_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def transform_to_matrix(transform):
    trans = transform.transform.translation
    rot = transform.transform.rotation
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = quaternion_to_matrix([rot.x, rot.y, rot.z, rot.w])
    mat[:3, 3] = [trans.x, trans.y, trans.z]
    return mat


class LatestRgbd:
    def __init__(self, color_topic, depth_topic, color_info_topic, queue_size, sync_slop):
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.color_msg = None
        self.depth_msg = None
        self.color_info_msg = None
        self.color_info_sub = rospy.Subscriber(color_info_topic, CameraInfo, self._info_cb, queue_size=1)
        self.color_sub = message_filters.Subscriber(color_topic, Image)
        self.depth_sub = message_filters.Subscriber(depth_topic, Image)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=queue_size,
            slop=sync_slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self._image_cb)

    def _info_cb(self, msg):
        with self.lock:
            self.color_info_msg = msg

    def _image_cb(self, color_msg, depth_msg):
        with self.lock:
            self.color_msg = color_msg
            self.depth_msg = depth_msg
        self.event.set()

    def wait(self, timeout):
        if not self.event.wait(timeout):
            raise RuntimeError("Timed out waiting for synchronized RGB-D images")
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while not rospy.is_shutdown():
            with self.lock:
                if self.color_msg is not None and self.depth_msg is not None and self.color_info_msg is not None:
                    color = self.bridge.imgmsg_to_cv2(self.color_msg, desired_encoding="bgr8")
                    depth = self.bridge.imgmsg_to_cv2(self.depth_msg, desired_encoding="passthrough")
                    return self.color_msg, color, depth, self.color_info_msg
            if rospy.Time.now() > deadline:
                raise RuntimeError("Timed out waiting for camera info")
            rospy.sleep(0.02)
        raise RuntimeError("ROS shutdown while waiting for RGB-D images")


class ChipsCanVisualServoGrasp:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.target_frame = rospy.get_param("~target_frame", "base_link")
        self.camera_frame = rospy.get_param("~camera_frame", "camera_color_optical_frame")
        self.eef_link = rospy.get_param("~eef_link", "gripper_center_link")
        self.move_group_name = rospy.get_param("~move_group", "manipulator")
        self.color_topic = rospy.get_param("~color_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.color_info_topic = rospy.get_param("~color_info_topic", "/camera/color/camera_info")
        self.depth_scale = float(rospy.get_param("~depth_scale", 0.001))
        self.min_depth_m = float(rospy.get_param("~min_depth_m", 0.2))
        self.max_depth_m = float(rospy.get_param("~max_depth_m", 1.0))
        self.depth_window_m = float(rospy.get_param("~depth_window_m", 0.08))
        self.target_clearance_m = float(rospy.get_param("~target_clearance_m", 0.005))
        self.xy_tolerance_m = float(rospy.get_param("~xy_tolerance_m", 0.008))
        self.z_tolerance_m = float(rospy.get_param("~z_tolerance_m", 0.008))
        self.align_orientation = bool(rospy.get_param("~align_orientation", False))
        self.orientation_tolerance_rad = math.radians(float(rospy.get_param("~orientation_tolerance_deg", 8.0)))
        self.orientation_xy_gate_m = float(rospy.get_param("~orientation_xy_gate_m", 0.020))
        self.orientation_z_gate_m = float(rospy.get_param("~orientation_z_gate_m", 0.020))
        self.freeze_visual_target = bool(rospy.get_param("~freeze_visual_target", False))
        self.single_shot_target = bool(rospy.get_param("~single_shot_target", False))
        self.target_offset_x_m = float(rospy.get_param("~target_offset_x_m", 0.0))
        self.target_offset_y_m = float(rospy.get_param("~target_offset_y_m", 0.0))
        self.apply_offset_after_align = bool(rospy.get_param("~apply_offset_after_align", False))
        self.rotate_after_align = bool(rospy.get_param("~rotate_after_align", False))
        self.descend_after_align = bool(rospy.get_param("~descend_after_align", False))
        self.final_target_clearance_m = float(rospy.get_param("~final_target_clearance_m", 0.035))
        self.max_descend_m = float(rospy.get_param("~max_descend_m", 0.10))
        self.max_step_xy_m = float(rospy.get_param("~max_step_xy_m", 0.01))
        self.max_step_z_m = float(rospy.get_param("~max_step_z_m", 0.01))
        self.max_iterations = int(rospy.get_param("~max_iterations", 12))
        self.eef_step = float(rospy.get_param("~eef_step", 0.002))
        self.min_fraction = float(rospy.get_param("~min_fraction", 0.95))
        self.avoid_collisions = bool(rospy.get_param("~avoid_collisions", True))
        self.velocity_scaling = float(rospy.get_param("~velocity_scaling", 0.02))
        self.accel_scaling = float(rospy.get_param("~accel_scaling", 0.02))
        self.execute = bool(rospy.get_param("~execute", False))
        self.confirm_execute = rospy.get_param("~confirm_execute", "")
        self.stop_before_close = bool(rospy.get_param("~stop_before_close", False))
        self.open_width_m = float(rospy.get_param("~open_width_m", 0.095))
        self.close_widths = [
            float(v.strip())
            for v in rospy.get_param("~close_widths", "0.085,0.080,0.075,0.070").split(",")
            if v.strip()
        ]
        self.force = float(rospy.get_param("~force", 25.0))
        self.speed = float(rospy.get_param("~speed", 20.0))
        self.lift_m = float(rospy.get_param("~lift_m", 0.01))
        self.required_grip_states = [
            int(v.strip())
            for v in str(rospy.get_param("~required_grip_states", rospy.get_param("~required_grip_state", 2))).split(",")
            if v.strip()
        ]

        if self.execute and self.confirm_execute != CONFIRM_TOKEN:
            raise RuntimeError("Refusing execution: confirm_execute must equal %s" % CONFIRM_TOKEN)
        if self.max_step_xy_m <= 0.0 or self.max_step_xy_m > 0.025:
            raise ValueError("max_step_xy_m must be in (0, 0.025]")
        if self.max_step_z_m <= 0.0 or self.max_step_z_m > 0.025:
            raise ValueError("max_step_z_m must be in (0, 0.025]")
        if self.lift_m < 0.0 or self.lift_m > 0.03:
            raise ValueError("lift_m must be in [0, 0.03]")
        if self.final_target_clearance_m < -0.06 or self.final_target_clearance_m > 0.12:
            raise ValueError("final_target_clearance_m must be in [-0.06, 0.12]")
        if self.max_descend_m <= 0.0 or self.max_descend_m > 0.15:
            raise ValueError("max_descend_m must be in (0, 0.15]")

        self.last_center_xy = None
        self.last_top_z = None
        self.last_can_axis = None
        self.frozen_center_xy = None
        self.frozen_top_z = None
        self.frozen_can_axis = None

        self.rgbd = LatestRgbd(
            self.color_topic,
            self.depth_topic,
            self.color_info_topic,
            queue_size=int(rospy.get_param("~queue_size", 10)),
            sync_slop=float(rospy.get_param("~sync_slop", 0.08)),
        )
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(0.5)

        self.display_pub = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=1,
            latch=True,
        )
        self.gripper_pub = rospy.Publisher("/gripper/ctrl", GripperCtrl, queue_size=1, latch=True)

        self.clear_scene_objects()
        self.move_group = moveit_commander.MoveGroupCommander(self.move_group_name)
        self.move_group.set_pose_reference_frame(self.target_frame)
        self.move_group.set_end_effector_link(self.eef_link)
        self.move_group.set_start_state_to_current_state()
        self.move_group.set_max_velocity_scaling_factor(self.velocity_scaling)
        self.move_group.set_max_acceleration_scaling_factor(self.accel_scaling)
        self.move_group.allow_replanning(False)

        rospy.logwarn("Visual-servo grasp: small Cartesian corrections only, then guarded gripper close.")
        if not self.execute:
            rospy.logwarn("DRY RUN: set execute:=true and confirm_execute:=%s to move robot.", CONFIRM_TOKEN)

    def clear_scene_objects(self):
        scene = moveit_commander.PlanningSceneInterface(synchronous=True)
        for name in ("demo_table", "chips_can_preview_box"):
            scene.remove_world_object(name)
        rospy.sleep(0.2)

    def segment_can(self, rgb, depth_m):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        green = (
            (hsv[:, :, 0] >= 35)
            & (hsv[:, :, 0] <= 95)
            & (hsv[:, :, 1] >= 35)
            & (hsv[:, :, 2] >= 35)
        )
        red = (
            ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 165))
            & (hsv[:, :, 1] >= 45)
            & (hsv[:, :, 2] >= 45)
        )
        yellow = (
            (hsv[:, :, 0] >= 12)
            & (hsv[:, :, 0] <= 34)
            & (hsv[:, :, 1] >= 45)
            & (hsv[:, :, 2] >= 50)
        )
        mask = (green | red | yellow) & (depth_m > self.min_depth_m) & (depth_m < self.max_depth_m)
        mask_u8 = mask.astype(np.uint8) * 255
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
        candidates = []
        for idx in range(1, n):
            x, y, w, h, area = stats[idx]
            if area >= 1000:
                candidates.append((area, idx, (x, y, w, h)))
        if not candidates:
            raise RuntimeError("No chips-can color component found")
        area, idx, bbox = max(candidates)
        selected = labels == idx
        z = depth_m[selected]
        if z.size < 500:
            raise RuntimeError("Not enough valid depth in chips-can mask")
        median = float(np.median(z))
        selected &= depth_m > median - self.depth_window_m
        selected &= depth_m < median + self.depth_window_m
        if int(selected.sum()) < 500:
            raise RuntimeError("Not enough depth points after median filter")
        return selected, bbox, median

    def estimate_can(self):
        color_msg, rgb, depth, info = self.rgbd.wait(timeout=3.0)
        depth_m = depth.astype(np.float32) * self.depth_scale
        mask, bbox, median_depth = self.segment_can(rgb, depth_m)
        rows, cols = np.where(mask)
        k = info.K
        fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
        z = depth_m[rows, cols].astype(np.float64)
        x = (cols.astype(np.float64) - cx) * z / fx
        y = (rows.astype(np.float64) - cy) * z / fy
        pts_camera = np.column_stack([x, y, z, np.ones_like(z)])

        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                color_msg.header.stamp,
                rospy.Duration(0.5),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            rospy.logwarn("Exact camera TF failed, using latest TF: %s", exc)
            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                rospy.Time(0),
                rospy.Duration(0.5),
            )
        base_t_camera = transform_to_matrix(tf)
        pts_base = (base_t_camera @ pts_camera.T).T[:, :3]
        xy = pts_base[:, :2]
        lo = np.percentile(xy, 5, axis=0)
        hi = np.percentile(xy, 95, axis=0)
        keep = (xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1])
        xy_keep = xy[keep]
        center_xy = xy_keep.mean(axis=0)
        centered_xy = xy_keep - center_xy
        _, _, vh = np.linalg.svd(centered_xy, full_matrices=False)
        can_axis = np.array([vh[0, 0], vh[0, 1], 0.0], dtype=np.float64)
        if can_axis[0] < 0.0:
            can_axis = -can_axis
        can_axis = normalize(can_axis)
        visible_top_z = float(np.percentile(pts_base[:, 2], 98.0))
        extents = np.ptp(pts_base[keep], axis=0)
        rospy.loginfo(
            "Visual estimate: bbox=%s mask=%d median_depth=%.3f center_xy=[%.4f, %.4f] can_axis=[%.4f, %.4f] top_z=%.4f extents=[%.3f, %.3f, %.3f]",
            bbox,
            int(mask.sum()),
            median_depth,
            center_xy[0],
            center_xy[1],
            can_axis[0],
            can_axis[1],
            visible_top_z,
            extents[0],
            extents[1],
            extents[2],
        )
        return center_xy, visible_top_z, can_axis

    def retime(self, trajectory):
        try:
            return self.move_group.retime_trajectory(
                self.move_group.get_current_state(),
                trajectory,
                velocity_scaling_factor=self.velocity_scaling,
                acceleration_scaling_factor=self.accel_scaling,
            )
        except TypeError:
            return self.move_group.retime_trajectory(
                self.move_group.get_current_state(),
                trajectory,
                self.velocity_scaling,
            )

    def cartesian_delta(self, dx, dy, dz, description, target_orientation=None):
        current = self.move_group.get_current_pose(self.eef_link).pose
        target = copy.deepcopy(current)
        target.position.x += dx
        target.position.y += dy
        target.position.z += dz
        if target_orientation is not None:
            target.orientation.x = float(target_orientation[0])
            target.orientation.y = float(target_orientation[1])
            target.orientation.z = float(target_orientation[2])
            target.orientation.w = float(target_orientation[3])
        self.move_group.set_start_state_to_current_state()
        trajectory, fraction = self.move_group.compute_cartesian_path(
            [target],
            self.eef_step,
            self.avoid_collisions,
        )
        rospy.loginfo(
            "%s: current=[%.4f, %.4f, %.4f] delta=[%.4f, %.4f, %.4f] fraction=%.4f",
            description,
            current.position.x,
            current.position.y,
            current.position.z,
            dx,
            dy,
            dz,
            fraction,
        )
        if fraction < self.min_fraction or not trajectory.joint_trajectory.points:
            raise RuntimeError("%s Cartesian path failed: fraction %.4f" % (description, fraction))
        trajectory = self.retime(trajectory)
        display = moveit_msgs.msg.DisplayTrajectory()
        display.model_id = "ur5e"
        display.trajectory_start = self.move_group.get_current_state()
        display.trajectory.append(trajectory)
        self.display_pub.publish(display)
        if not self.execute:
            rospy.logwarn("DRY RUN: not executing %s", description)
            return
        rospy.logwarn("EXECUTING: %s", description)
        ok = self.move_group.execute(trajectory, wait=True)
        self.move_group.stop()
        if not ok:
            raise RuntimeError("%s execution failed" % description)

    def plan_pose_target(self, target, description):
        self.move_group.set_start_state_to_current_state()
        self.move_group.set_pose_target(target, self.eef_link)
        result = self.move_group.plan()
        if isinstance(result, tuple):
            success = bool(result[0])
            trajectory = result[1]
        else:
            trajectory = result
            success = bool(trajectory and trajectory.joint_trajectory.points)
        self.move_group.clear_pose_targets()
        if not success or not trajectory.joint_trajectory.points:
            raise RuntimeError("%s planning failed" % description)
        trajectory = self.retime(trajectory)
        display = moveit_msgs.msg.DisplayTrajectory()
        display.model_id = "ur5e"
        display.trajectory_start = self.move_group.get_current_state()
        display.trajectory.append(trajectory)
        self.display_pub.publish(display)
        rospy.loginfo("%s: planned %d trajectory points", description, len(trajectory.joint_trajectory.points))
        if not self.execute:
            rospy.logwarn("DRY RUN: not executing %s", description)
            return
        rospy.logwarn("EXECUTING: %s", description)
        ok = self.move_group.execute(trajectory, wait=True)
        self.move_group.stop()
        if not ok:
            raise RuntimeError("%s execution failed" % description)

    def wait_gripper_state(self, timeout=2.0):
        try:
            return rospy.wait_for_message("/gripper/states", GripperState, timeout=timeout)
        except rospy.ROSException:
            return None

    def command_width(self, width_m, description):
        position = width_to_driver_position(width_m)
        state = self.wait_gripper_state()
        if state is not None:
            rospy.loginfo(
                "Before %s: grip_state=%d, position=%.1f, width~=%.1f mm",
                description,
                state.grip_state,
                state.position,
                driver_position_to_width(state.position) * 1000.0,
            )
        rospy.loginfo("%s: width %.1f mm -> position=%d", description, width_m * 1000.0, position)
        if not self.execute:
            rospy.logwarn("DRY RUN: not publishing gripper command")
            return state
        msg = GripperCtrl()
        msg.initialize = False
        msg.position = float(position)
        msg.force = self.force
        msg.speed = self.speed
        self.gripper_pub.publish(msg)
        rospy.sleep(1.0)
        state = self.wait_gripper_state()
        if state is not None:
            rospy.loginfo(
                "After %s: grip_state=%d, position=%.1f, width~=%.1f mm, target_position=%.1f",
                description,
                state.grip_state,
                state.position,
                driver_position_to_width(state.position) * 1000.0,
                state.target_position,
            )
        return state

    def verify_grasp_before_lift(self):
        state = self.wait_gripper_state()
        if state is None:
            raise RuntimeError("Refusing lift: no /gripper/states received")
        rospy.loginfo(
            "Grip check before lift: grip_state=%d, position=%.1f, width~=%.1f mm",
            state.grip_state,
            state.position,
            driver_position_to_width(state.position) * 1000.0,
        )
        if int(state.grip_state) not in self.required_grip_states:
            raise RuntimeError(
                "Refusing lift: grip_state=%d, expected one of %s. The gripper likely did not contact the can."
                % (state.grip_state, self.required_grip_states)
            )

    def target_orientation_for_can(self, can_axis, current_pose):
        jaw_axis = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float64), can_axis)
        if np.linalg.norm(jaw_axis) < 1e-6:
            jaw_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        jaw_axis = normalize(jaw_axis)

        current_q = [
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w,
        ]
        current_rot = quaternion_to_matrix(current_q)
        current_y = current_rot[:, 1]
        current_y[2] = 0.0
        if np.linalg.norm(current_y) >= 1e-6:
            current_y = normalize(current_y)
            if float(np.dot(jaw_axis, current_y)) < 0.0:
                jaw_axis = -jaw_axis

        desired_rot = rotation_from_x_down(jaw_axis)
        desired_q = quaternion_from_matrix(desired_rot)
        current_y = current_rot[:, 1] - np.dot(current_rot[:, 1], np.array([0.0, 0.0, 1.0])) * np.array([0.0, 0.0, 1.0])
        if np.linalg.norm(current_y) < 1e-6:
            angle_error = math.pi
        else:
            current_y = normalize(current_y)
            angle_error = math.acos(max(-1.0, min(1.0, abs(float(np.dot(jaw_axis, current_y))))))
        return desired_q, angle_error, jaw_axis

    def servo_until_aligned(self):
        for iteration in range(1, self.max_iterations + 1):
            observed_center_xy, observed_top_z, observed_can_axis = self.estimate_can()
            if self.freeze_visual_target and self.frozen_center_xy is None:
                self.frozen_center_xy = observed_center_xy.copy()
                if not self.apply_offset_after_align:
                    self.frozen_center_xy += np.array([self.target_offset_x_m, self.target_offset_y_m], dtype=np.float64)
                self.frozen_top_z = float(observed_top_z)
                self.frozen_can_axis = observed_can_axis.copy()
                rospy.logwarn(
                    "Frozen high-view target: center_xy=[%.4f, %.4f] offset=[%.4f, %.4f] top_z=%.4f can_axis=[%.4f, %.4f]",
                    self.frozen_center_xy[0],
                    self.frozen_center_xy[1],
                    self.target_offset_x_m,
                    self.target_offset_y_m,
                    self.frozen_top_z,
                    self.frozen_can_axis[0],
                    self.frozen_can_axis[1],
                )
            if self.freeze_visual_target:
                center_xy = self.frozen_center_xy
                top_z = self.frozen_top_z
                can_axis = self.frozen_can_axis
            else:
                center_xy = observed_center_xy
                top_z = observed_top_z
                can_axis = observed_can_axis
            self.last_center_xy = center_xy
            self.last_top_z = top_z
            self.last_can_axis = can_axis
            current = self.move_group.get_current_pose(self.eef_link).pose
            target_z = top_z + self.target_clearance_m
            err_x = float(center_xy[0] - current.position.x)
            err_y = float(center_xy[1] - current.position.y)
            err_z = float(target_z - current.position.z)
            target_orientation = None
            angle_error = 0.0
            jaw_axis = None
            position_close_for_orientation = (
                abs(err_x) <= self.orientation_xy_gate_m
                and abs(err_y) <= self.orientation_xy_gate_m
                and abs(err_z) <= self.orientation_z_gate_m
            )
            if self.align_orientation and position_close_for_orientation:
                target_orientation, angle_error, jaw_axis = self.target_orientation_for_can(can_axis, current)

            if jaw_axis is None:
                rospy.loginfo(
                    "Servo iteration %d error=[%.4f, %.4f, %.4f], target_z=%.4f",
                    iteration,
                    err_x,
                    err_y,
                    err_z,
                    target_z,
                )
            else:
                rospy.loginfo(
                    "Servo iteration %d error=[%.4f, %.4f, %.4f], yaw_error=%.1f deg, jaw_axis=[%.4f, %.4f], target_z=%.4f",
                    iteration,
                    err_x,
                    err_y,
                    err_z,
                    math.degrees(angle_error),
                    jaw_axis[0],
                    jaw_axis[1],
                    target_z,
                )
            position_aligned = (
                abs(err_x) <= self.xy_tolerance_m
                and abs(err_y) <= self.xy_tolerance_m
                and abs(err_z) <= self.z_tolerance_m
            )
            orientation_aligned = (not self.align_orientation) or angle_error <= self.orientation_tolerance_rad
            if position_aligned and orientation_aligned:
                rospy.loginfo("Visual-servo alignment reached.")
                return
            dx = max(-self.max_step_xy_m, min(self.max_step_xy_m, err_x))
            dy = max(-self.max_step_xy_m, min(self.max_step_xy_m, err_y))
            dz = max(-self.max_step_z_m, min(self.max_step_z_m, err_z))
            self.cartesian_delta(dx, dy, dz, "visual servo correction %d" % iteration, target_orientation)
        raise RuntimeError("Visual servo did not converge within %d iterations" % self.max_iterations)

    def move_to_single_shot_target(self):
        center_xy, top_z, can_axis = self.estimate_can()
        if not self.apply_offset_after_align:
            center_xy = center_xy + np.array([self.target_offset_x_m, self.target_offset_y_m], dtype=np.float64)
        self.last_center_xy = center_xy
        self.last_top_z = top_z
        self.last_can_axis = can_axis
        current = self.move_group.get_current_pose(self.eef_link).pose
        target = copy.deepcopy(current)
        target.position.x = float(center_xy[0])
        target.position.y = float(center_xy[1])
        target.position.z = float(top_z + self.target_clearance_m)
        if self.align_orientation:
            target_q, angle_error, jaw_axis = self.target_orientation_for_can(can_axis, current)
            target.orientation.x = float(target_q[0])
            target.orientation.y = float(target_q[1])
            target.orientation.z = float(target_q[2])
            target.orientation.w = float(target_q[3])
            rospy.loginfo(
                "Single-shot target: center_xy=[%.4f, %.4f] top_z=%.4f target_z=%.4f can_axis=[%.4f, %.4f] jaw_axis=[%.4f, %.4f] yaw_error=%.1f deg",
                center_xy[0],
                center_xy[1],
                top_z,
                target.position.z,
                can_axis[0],
                can_axis[1],
                jaw_axis[0],
                jaw_axis[1],
                math.degrees(angle_error),
            )
        else:
            rospy.loginfo(
                "Single-shot target: center_xy=[%.4f, %.4f] top_z=%.4f target_z=%.4f",
                center_xy[0],
                center_xy[1],
                top_z,
                target.position.z,
            )
        self.plan_pose_target(target, "move to frozen high target")

    def apply_calibrated_offset(self):
        if abs(self.target_offset_x_m) < 1e-6 and abs(self.target_offset_y_m) < 1e-6:
            rospy.loginfo("No calibrated XY offset requested.")
            return
        rospy.loginfo(
            "Applying calibrated XY offset at grasp orientation: dx=%.4f dy=%.4f",
            self.target_offset_x_m,
            self.target_offset_y_m,
        )
        self.cartesian_delta(
            self.target_offset_x_m,
            self.target_offset_y_m,
            0.0,
            "apply calibrated XY offset",
        )

    def rotate_to_grasp_orientation(self):
        if self.last_can_axis is None:
            raise RuntimeError("Cannot rotate: no can_axis estimate is available")
        current = self.move_group.get_current_pose(self.eef_link).pose
        target_q, angle_error, jaw_axis = self.target_orientation_for_can(self.last_can_axis, current)
        if angle_error <= self.orientation_tolerance_rad:
            rospy.loginfo("Gripper orientation already aligned: yaw_error=%.1f deg", math.degrees(angle_error))
            return
        target = copy.deepcopy(current)
        target.orientation.x = float(target_q[0])
        target.orientation.y = float(target_q[1])
        target.orientation.z = float(target_q[2])
        target.orientation.w = float(target_q[3])
        rospy.loginfo(
            "Rotate gripper at high target: yaw_error=%.1f deg, jaw_axis=[%.4f, %.4f]",
            math.degrees(angle_error),
            jaw_axis[0],
            jaw_axis[1],
        )
        self.plan_pose_target(target, "rotate gripper at frozen high target")

    def descend_to_grasp_height(self):
        if self.last_top_z is None:
            raise RuntimeError("Cannot descend: no visual top_z estimate is available")
        current = self.move_group.get_current_pose(self.eef_link).pose
        target_z = float(self.last_top_z + self.final_target_clearance_m)
        dz = target_z - float(current.position.z)
        descend = -dz
        rospy.loginfo(
            "Computed grasp descent: current_z=%.4f top_z=%.4f final_clearance=%.4f target_z=%.4f dz=%.4f",
            current.position.z,
            self.last_top_z,
            self.final_target_clearance_m,
            target_z,
            dz,
        )
        if dz >= -1e-4:
            raise RuntimeError("Refusing descend: target_z is not below current_z")
        if descend > self.max_descend_m:
            raise RuntimeError(
                "Refusing descend: %.4f m exceeds max_descend_m %.4f. Increase only after verifying clearance."
                % (descend, self.max_descend_m)
            )
        self.cartesian_delta(0.0, 0.0, dz, "computed vertical descend")

    def run(self):
        self.command_width(self.open_width_m, "open gripper")
        if self.single_shot_target:
            self.move_to_single_shot_target()
        else:
            self.servo_until_aligned()
        if self.rotate_after_align:
            self.rotate_to_grasp_orientation()
        if self.apply_offset_after_align:
            self.apply_calibrated_offset()
        if self.descend_after_align:
            self.descend_to_grasp_height()
        if self.stop_before_close:
            rospy.logwarn("Stopping before close as requested.")
            return
        for width in self.close_widths:
            self.command_width(width, "close gripper")
        self.verify_grasp_before_lift()
        if self.lift_m > 1e-6:
            self.cartesian_delta(0.0, 0.0, self.lift_m, "lift test")
        rospy.loginfo("Visual-servo grasp sequence finished.")


def main():
    rospy.init_node("demo_chips_can_visual_servo_grasp")
    ChipsCanVisualServoGrasp().run()


if __name__ == "__main__":
    main()
