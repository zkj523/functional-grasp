#!/usr/bin/env python3

import argparse
import json
import math
import os
import sys
import threading
import time


def _find_system_libffi():
    for candidate in (
        "/lib/x86_64-linux-gnu/libffi.so.7",
        "/usr/lib/x86_64-linux-gnu/libffi.so.7",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def _maybe_reexec_with_system_libffi():
    if os.environ.get("_SINGLE_RGBD_EE_CAPTURE_LIBFFI_READY") == "1":
        return

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    conda_libffi = os.path.join(conda_prefix, "lib", "libffi.so.7")
    if not os.path.exists(conda_libffi):
        return

    conda_libffi_target = os.path.realpath(conda_libffi)
    if os.path.basename(conda_libffi_target).startswith("libffi.so.7"):
        return

    system_libffi = _find_system_libffi()
    if system_libffi is None:
        return

    environment = os.environ.copy()
    preload_entries = [
        entry for entry in environment.get("LD_PRELOAD", "").split(":") if entry
    ]
    if system_libffi not in preload_entries:
        environment["LD_PRELOAD"] = ":".join([system_libffi] + preload_entries)
    environment["_SINGLE_RGBD_EE_CAPTURE_LIBFFI_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, environment)


_maybe_reexec_with_system_libffi()

import cv2
import message_filters
import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image


def _ensure_ros_log_dir(node_name):
    configured_log_dir = os.environ.get("ROS_LOG_DIR")
    if configured_log_dir:
        candidate_log_dir = os.path.abspath(os.path.expanduser(configured_log_dir))
    else:
        ros_home = os.environ.get("ROS_HOME", os.path.expanduser("~/.ros"))
        candidate_log_dir = os.path.join(
            os.path.abspath(os.path.expanduser(ros_home)), "log"
        )

    if _is_writable_directory(candidate_log_dir):
        return

    fallback_log_dir = os.path.join("/tmp", "ros_log", node_name)
    os.makedirs(fallback_log_dir, exist_ok=True)
    os.environ["ROS_LOG_DIR"] = fallback_log_dir


def _is_writable_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
        probe_file = os.path.join(path, ".write_probe")
        with open(probe_file, "w", encoding="utf-8"):
            pass
        os.remove(probe_file)
        return True
    except OSError:
        return False


def _time_to_dict(stamp):
    return {
        "secs": int(stamp.secs),
        "nsecs": int(stamp.nsecs),
        "time_sec": float(stamp.to_sec()),
    }


def _camera_info_to_dict(msg):
    if msg is None:
        return None
    return {
        "frame_id": msg.header.frame_id,
        "stamp": _time_to_dict(msg.header.stamp),
        "height": int(msg.height),
        "width": int(msg.width),
        "distortion_model": msg.distortion_model,
        "D": [float(x) for x in msg.D],
        "K": [float(x) for x in msg.K],
        "R": [float(x) for x in msg.R],
        "P": [float(x) for x in msg.P],
    }


def _quaternion_to_rpy(x, y, z, w):
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return {"roll": roll, "pitch": pitch, "yaw": yaw}


def _squeeze_depth(depth_image):
    if depth_image.ndim == 3 and depth_image.shape[2] == 1:
        return depth_image.squeeze(axis=2)
    return depth_image


def _make_depth_preview(depth_image):
    depth = _squeeze_depth(depth_image)
    if depth.ndim != 2:
        return None

    finite_mask = np.isfinite(depth)
    if np.issubdtype(depth.dtype, np.integer):
        valid_mask = finite_mask & (depth > 0)
    elif np.issubdtype(depth.dtype, np.floating):
        valid_mask = finite_mask & (depth > 0.0)
    else:
        valid_mask = finite_mask

    preview_gray = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid_mask):
        valid_depth = depth[valid_mask].astype(np.float32)
        depth_min = float(np.percentile(valid_depth, 1.0))
        depth_max = float(np.percentile(valid_depth, 99.0))
        if depth_max <= depth_min:
            depth_min = float(valid_depth.min())
            depth_max = float(valid_depth.max())
        if depth_max > depth_min:
            scaled = (valid_depth - depth_min) / (depth_max - depth_min)
            scaled = np.clip(scaled * 255.0, 0.0, 255.0)
            preview_gray[valid_mask] = scaled.astype(np.uint8)

    preview = cv2.applyColorMap(preview_gray, cv2.COLORMAP_JET)
    preview[~valid_mask] = 0
    return preview


def _unique_prefix(output_dir, prefix, stamp_str):
    base = os.path.join(output_dir, f"{prefix}_{stamp_str}")
    candidate = base
    suffix = 1
    while os.path.exists(candidate + "_metadata.json"):
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


class SingleRgbdEeCapture:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()
        self.event = threading.Event()
        self.lock = threading.Lock()
        self.color_msg = None
        self.depth_msg = None

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(args.tf_cache_sec))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        if args.tf_warmup > 0.0:
            rospy.sleep(args.tf_warmup)

        self.color_sub = message_filters.Subscriber(args.color_topic, Image)
        self.depth_sub = message_filters.Subscriber(args.depth_topic, Image)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=args.queue_size,
            slop=args.sync_slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self._synced_callback)

    def _synced_callback(self, color_msg, depth_msg):
        with self.lock:
            self.color_msg = color_msg
            self.depth_msg = depth_msg
        self.event.set()

    def snapshot_latest_images(self):
        with self.lock:
            if self.color_msg is None or self.depth_msg is None:
                return None, None
            return self.color_msg, self.depth_msg

    def wait_for_images(self):
        deadline = time.monotonic() + self.args.timeout
        while not rospy.is_shutdown():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    "Timed out waiting for synchronized RGB and depth images."
                )
            if self.event.wait(min(0.1, remaining)):
                with self.lock:
                    return self.color_msg, self.depth_msg
        raise RuntimeError("ROS shutdown before images were captured.")

    def lookup_pose(self, reference_frame, child_frame, stamp):
        exact_exception = None
        try:
            return self._lookup_transform(reference_frame, child_frame, stamp), "image_timestamp"
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            exact_exception = exc

        if not self.args.use_latest_tf:
            raise exact_exception

        rospy.logwarn(
            "Exact TF lookup failed at image timestamp; using latest available TF. "
            f"Reason: {exact_exception}"
        )
        return self._lookup_transform(reference_frame, child_frame, rospy.Time(0)), "latest_available"

    def lookup_ee_pose(self, stamp):
        return self.lookup_pose(self.args.base_frame, self.args.ee_frame, stamp)

    def lookup_camera_pose(self, stamp, color_frame_id):
        camera_frame = self.args.camera_frame or color_frame_id
        return self.lookup_pose(self.args.base_frame, camera_frame, stamp)

    def _lookup_transform(self, reference_frame, child_frame, stamp):
        deadline = rospy.Time.now() + rospy.Duration(self.args.tf_timeout)
        last_exception = None
        while not rospy.is_shutdown():
            try:
                return self.tf_buffer.lookup_transform(
                    reference_frame,
                    child_frame,
                    stamp,
                    rospy.Duration(0.1),
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as exc:
                last_exception = exc
                if rospy.Time.now() >= deadline:
                    raise last_exception
                rospy.sleep(0.05)
        raise RuntimeError("ROS shutdown during TF lookup.")


def _transform_to_pose_dict(reference_frame, child_frame, transform, lookup_mode):
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    rpy = _quaternion_to_rpy(rotation.x, rotation.y, rotation.z, rotation.w)
    return {
        "reference_frame": reference_frame,
        "child_frame": child_frame,
        "lookup_mode": lookup_mode,
        "transform_stamp": _time_to_dict(transform.header.stamp),
        "position_m": {
            "x": float(translation.x),
            "y": float(translation.y),
            "z": float(translation.z),
        },
        "orientation_xyzw": {
            "x": float(rotation.x),
            "y": float(rotation.y),
            "z": float(rotation.z),
            "w": float(rotation.w),
        },
        "orientation_rpy_rad": rpy,
    }


def _lookup_optional_camera_pose(args, capture, color_msg):
    camera_frame = args.camera_frame or color_msg.header.frame_id
    if not camera_frame:
        return None
    try:
        transform, lookup_mode = capture.lookup_camera_pose(
            color_msg.header.stamp, color_msg.header.frame_id
        )
        return _transform_to_pose_dict(
            args.base_frame, camera_frame, transform, lookup_mode
        )
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
        RuntimeError,
    ) as exc:
        rospy.logwarn(
            "Camera pose lookup failed for %s -> %s: %s",
            args.base_frame,
            camera_frame,
            exc,
        )
        return None


def _draw_text_block(image, lines):
    annotated = image.copy()
    line_height = 24
    margin = 10
    box_height = margin * 2 + line_height * len(lines)
    box_width = min(annotated.shape[1], 760)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (box_width, box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)
    for index, line in enumerate(lines):
        cv2.putText(
            annotated,
            line,
            (margin, margin + 18 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (80, 255, 80),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _annotate_color_preview(color_image, color_msg, depth_msg, pose, status_text):
    stamp = color_msg.header.stamp
    stamp_str = f"{stamp.secs}_{stamp.nsecs:09d}"
    sync_delta_sec = abs((color_msg.header.stamp - depth_msg.header.stamp).to_sec())
    position = pose["position_m"]
    orientation = pose["orientation_xyzw"]
    lines = [
        "s: save current sample   q: quit",
        status_text,
        f"stamp: {stamp_str}  sync_delta: {sync_delta_sec:.6f}s",
        f"pose: {pose['reference_frame']} -> {pose['child_frame']} ({pose['lookup_mode']})",
        (
            "xyz[m]: "
            f"{position['x']:.4f}, {position['y']:.4f}, {position['z']:.4f}"
        ),
        (
            "quat xyzw: "
            f"{orientation['x']:.4f}, {orientation['y']:.4f}, "
            f"{orientation['z']:.4f}, {orientation['w']:.4f}"
        ),
    ]
    return _draw_text_block(color_image, lines)


def _save_capture(args, capture, color_msg, depth_msg, transform, lookup_mode):
    os.makedirs(args.out_dir, exist_ok=True)

    stamp = color_msg.header.stamp
    if stamp == rospy.Time(0):
        stamp = rospy.Time.now()
    stamp_str = f"{stamp.secs}_{stamp.nsecs:09d}"
    file_prefix = _unique_prefix(args.out_dir, args.prefix, stamp_str)

    color_image = CvBridge().imgmsg_to_cv2(color_msg, desired_encoding=args.color_encoding)
    depth_image = CvBridge().imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
    depth_image = _squeeze_depth(depth_image)

    color_path = file_prefix + "_rgb.png"
    depth_npy_path = file_prefix + "_depth.npy"
    depth_png_path = None
    depth_preview_path = file_prefix + "_depth_preview.png"
    metadata_path = file_prefix + "_metadata.json"

    if not cv2.imwrite(color_path, color_image):
        raise RuntimeError(f"Failed to write RGB image: {color_path}")

    np.save(depth_npy_path, depth_image)

    if depth_image.dtype in (np.uint8, np.uint16):
        depth_png_path = file_prefix + "_depth.png"
        if not cv2.imwrite(depth_png_path, depth_image):
            raise RuntimeError(f"Failed to write depth image: {depth_png_path}")

    preview = _make_depth_preview(depth_image)
    if preview is not None:
        cv2.imwrite(depth_preview_path, preview)
    else:
        depth_preview_path = None

    color_info = _wait_for_camera_info(
        args.color_info_topic, args.camera_info_timeout
    )
    depth_info = _wait_for_camera_info(
        args.depth_info_topic, args.camera_info_timeout
    )

    sync_delta_sec = abs((color_msg.header.stamp - depth_msg.header.stamp).to_sec())
    pose = _transform_to_pose_dict(
        args.base_frame, args.ee_frame, transform, lookup_mode
    )
    camera_pose = _lookup_optional_camera_pose(args, capture, color_msg)

    metadata = {
        "stamp": _time_to_dict(stamp),
        "color_stamp": _time_to_dict(color_msg.header.stamp),
        "depth_stamp": _time_to_dict(depth_msg.header.stamp),
        "sync_delta_sec": sync_delta_sec,
        "topics": {
            "color": args.color_topic,
            "depth": args.depth_topic,
            "color_camera_info": args.color_info_topic,
            "depth_camera_info": args.depth_info_topic,
        },
        "frames": {
            "color_frame_id": color_msg.header.frame_id,
            "depth_frame_id": depth_msg.header.frame_id,
            "base_frame": args.base_frame,
            "ee_frame": args.ee_frame,
            "camera_frame": args.camera_frame or color_msg.header.frame_id,
        },
        "images": {
            "color_encoding": color_msg.encoding,
            "depth_encoding": depth_msg.encoding,
            "color_shape": list(color_image.shape),
            "depth_shape": list(depth_image.shape),
            "depth_dtype": str(depth_image.dtype),
        },
        "saved_files": {
            "rgb_png": os.path.basename(color_path),
            "depth_npy": os.path.basename(depth_npy_path),
            "depth_png": os.path.basename(depth_png_path) if depth_png_path else None,
            "depth_preview_png": (
                os.path.basename(depth_preview_path) if depth_preview_path else None
            ),
            "metadata_json": os.path.basename(metadata_path),
        },
        "pose": pose,
        "camera_pose": camera_pose,
        "camera_info": {
            "color": _camera_info_to_dict(color_info),
            "depth": _camera_info_to_dict(depth_info),
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2)

    return {
        "rgb": color_path,
        "depth_npy": depth_npy_path,
        "depth_png": depth_png_path,
        "depth_preview": depth_preview_path,
        "metadata": metadata_path,
        "pose": metadata["pose"],
    }


def _wait_for_camera_info(topic, timeout):
    if not topic or timeout <= 0.0:
        return None
    try:
        return rospy.wait_for_message(topic, CameraInfo, timeout=timeout)
    except rospy.ROSException:
        rospy.logwarn(f"CameraInfo not received from {topic} within {timeout:.2f}s.")
        return None


def _run_interactive(args, capture):
    rospy.loginfo("Interactive mode: press 's' in the RGB window to save, 'q' to quit.")
    rate = rospy.Rate(args.preview_rate)
    status_text = "waiting for synchronized RGB-D sample..."
    last_display_stamp = None

    while not rospy.is_shutdown():
        color_msg, depth_msg = capture.snapshot_latest_images()
        if color_msg is None or depth_msg is None:
            cv2.waitKey(1)
            rate.sleep()
            continue

        try:
            color_image = capture.bridge.imgmsg_to_cv2(
                color_msg, desired_encoding=args.color_encoding
            )
            depth_image = capture.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
            depth_image = _squeeze_depth(depth_image)
            transform, lookup_mode = capture.lookup_ee_pose(color_msg.header.stamp)
            pose = _transform_to_pose_dict(
                args.base_frame, args.ee_frame, transform, lookup_mode
            )
        except Exception as exc:
            rospy.logwarn_throttle(2.0, f"Preview update failed: {exc}")
            cv2.waitKey(1)
            rate.sleep()
            continue

        current_stamp = (color_msg.header.stamp.secs, color_msg.header.stamp.nsecs)
        if current_stamp != last_display_stamp:
            last_display_stamp = current_stamp
            status_text = "ready"

        annotated_color = _annotate_color_preview(
            color_image, color_msg, depth_msg, pose, status_text
        )
        depth_preview = _make_depth_preview(depth_image)

        cv2.imshow("RGB + end-effector pose", annotated_color)
        if depth_preview is not None:
            cv2.imshow("Aligned depth preview", depth_preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            try:
                result = _save_capture(
                    args, capture, color_msg, depth_msg, transform, lookup_mode
                )
                status_text = f"saved: {os.path.basename(result['metadata'])}"
                rospy.loginfo(f"Saved current sample metadata: {result['metadata']}")
            except Exception as exc:
                status_text = f"save failed: {exc}"
                rospy.logerr(status_text)

        rate.sleep()

    cv2.destroyAllWindows()
    rospy.signal_shutdown("interactive capture closed")
    return 0


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Capture one synchronized RGB-D frame and end-effector pose."
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.expanduser("~/rgbd_ee_captures"),
        help="Directory for saved capture files.",
    )
    parser.add_argument("--prefix", default="capture", help="Output file prefix.")
    parser.add_argument(
        "--color-topic",
        default="/camera/color/image_raw",
        help="RGB image topic.",
    )
    parser.add_argument(
        "--depth-topic",
        default="/camera/aligned_depth_to_color/image_raw",
        help="Depth image topic. Default is aligned to color.",
    )
    parser.add_argument(
        "--color-info-topic",
        default="/camera/color/camera_info",
        help="RGB CameraInfo topic saved into metadata.",
    )
    parser.add_argument(
        "--depth-info-topic",
        default="/camera/aligned_depth_to_color/camera_info",
        help="Depth CameraInfo topic saved into metadata.",
    )
    parser.add_argument(
        "--base-frame",
        default="base_link_inertia1",
        help="Reference frame for end-effector pose.",
    )
    parser.add_argument(
        "--ee-frame",
        default="gripper_center_link",
        help="End-effector frame. Use tool0 if you want the UR tool frame.",
    )
    parser.add_argument(
        "--color-encoding",
        default="bgr8",
        help="cv_bridge encoding used before saving RGB PNG.",
    )
    parser.add_argument(
        "--camera-frame",
        default="",
        help=(
            "Camera frame saved as camera_pose. Defaults to the RGB image "
            "message frame_id."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Show live RGB/depth preview. Press s to save one sample, q to quit.",
    )
    parser.add_argument(
        "--preview-rate",
        type=float,
        default=15.0,
        help="Interactive preview refresh rate in Hz.",
    )
    parser.add_argument(
        "--sync-slop",
        type=float,
        default=0.03,
        help="Allowed RGB/depth timestamp difference in seconds.",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=20,
        help="Message filter queue size.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for synchronized images.",
    )
    parser.add_argument(
        "--tf-timeout",
        type=float,
        default=2.0,
        help="Seconds to wait for TF lookup.",
    )
    parser.add_argument(
        "--tf-cache-sec",
        type=float,
        default=30.0,
        help="TF buffer cache length in seconds.",
    )
    parser.add_argument(
        "--tf-warmup",
        type=float,
        default=0.5,
        help="Seconds to fill the TF buffer before taking the image sample.",
    )
    parser.add_argument(
        "--camera-info-timeout",
        type=float,
        default=1.0,
        help="Seconds to wait for CameraInfo; set 0 to skip.",
    )
    parser.add_argument(
        "--no-latest-tf",
        dest="use_latest_tf",
        action="store_false",
        help="Fail instead of falling back to latest TF when exact image-time TF fails.",
    )
    parser.set_defaults(use_latest_tf=True)
    return parser.parse_args(rospy.myargv(argv=argv)[1:])


def main(argv=None):
    argv = argv or sys.argv
    args = _parse_args(argv)
    args.out_dir = os.path.abspath(os.path.expanduser(args.out_dir))

    _ensure_ros_log_dir("single_rgbd_ee_capture")
    rospy.init_node("single_rgbd_ee_capture", anonymous=True)

    rospy.loginfo(f"Waiting for RGB topic: {args.color_topic}")
    rospy.loginfo(f"Waiting for depth topic: {args.depth_topic}")
    rospy.loginfo(f"End-effector pose: {args.base_frame} -> {args.ee_frame}")

    capture = SingleRgbdEeCapture(args)
    if args.interactive:
        return _run_interactive(args, capture)

    color_msg, depth_msg = capture.wait_for_images()
    transform, lookup_mode = capture.lookup_ee_pose(color_msg.header.stamp)
    result = _save_capture(args, capture, color_msg, depth_msg, transform, lookup_mode)

    rospy.loginfo("Single RGB-D + EE pose capture complete.")
    rospy.loginfo(f"RGB: {result['rgb']}")
    rospy.loginfo(f"Depth NPY: {result['depth_npy']}")
    if result["depth_png"]:
        rospy.loginfo(f"Depth PNG: {result['depth_png']}")
    if result["depth_preview"]:
        rospy.loginfo(f"Depth preview: {result['depth_preview']}")
    rospy.loginfo(f"Metadata: {result['metadata']}")
    rospy.signal_shutdown("capture complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
