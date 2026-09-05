#!/usr/bin/env python3
"""Quality gate for a newly prepared real chips-can run."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


def quat_matrix(q):
    x, y, z, w = np.asarray(q, dtype=np.float64)
    x, y, z, w = (np.asarray([x, y, z, w]) / np.linalg.norm([x, y, z, w])).tolist()
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    reg_path = args.run_dir / "registration" / "registration.yaml"
    candidate_path = args.run_dir / "registration" / "ag95_plan_only_candidates_base.json"
    reg = yaml.safe_load(reg_path.read_text(encoding="utf-8"))
    candidate_data = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate = candidate_data["candidates"][0]

    metrics = reg["observed_to_model_distance_m"]
    p95 = float(metrics["p95"])
    table_rmse = float(reg["table_plane_rmse_m"])
    span = float(reg["observed_axis_span_m"])
    visible = int(reg["object_visible_points"])
    center = np.asarray(reg["object_center_base_m"], dtype=np.float64)
    axis = np.asarray(reg["object_axis_base"], dtype=np.float64)
    scene = candidate_data["planning_scene"]
    table_top = float(scene["table_pose_xyz_m"][2]) + 0.5 * float(scene["table_size_m"][2])
    anchor = np.asarray(candidate_data["execution_anchor_base_m"], dtype=np.float64)
    position = candidate["position"]
    high = np.asarray([position[k] for k in ("x", "y", "z")], dtype=np.float64)
    orientation = candidate["orientation"]
    rotation = quat_matrix([orientation[k] for k in ("x", "y", "z", "w")])
    approach_down_alignment = float(np.dot(rotation[:, 0], [0.0, 0.0, -1.0]))

    failures = []
    checks = {
        "registration_p95_m": p95,
        "table_rmse_m": table_rmse,
        "observed_axis_span_m": span,
        "visible_object_points": visible,
        "object_center_base_m": center.tolist(),
        "object_axis_base": axis.tolist(),
        "table_top_z_m": table_top,
        "execution_anchor_base_m": anchor.tolist(),
        "high_target_base_m": high.tolist(),
        "approach_down_alignment": approach_down_alignment,
    }
    if p95 > 0.015: failures.append("registration p95 exceeds 15 mm")
    if table_rmse > 0.004: failures.append("table-plane RMSE exceeds 4 mm")
    if not 0.18 <= span <= 0.28: failures.append("observed long-axis span outside [0.18, 0.28] m")
    if visible < 5000: failures.append("fewer than 5000 visible object points")
    if abs(float(axis[2])) > 0.12: failures.append("can is not sufficiently side-lying")
    if not (-0.45 <= center[0] <= 0.55 and 0.15 <= center[1] <= 0.85):
        failures.append("object center outside configured tabletop workspace")
    if not 0.17 <= high[2] - anchor[2] <= 0.23:
        failures.append("high target is not approximately 200 mm above anchor")
    if not 0.030 <= (anchor[2] - 0.035) - table_top <= 0.070:
        failures.append("computed final gripper-center table clearance is unsafe")
    if approach_down_alignment < 0.995:
        failures.append("candidate approach axis is not vertically downward")

    report = {"schema": "online_chips_can_quality_gate_v1", "passed": not failures,
              "checks": checks, "failures": failures}
    output = args.run_dir / "quality_gate.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
