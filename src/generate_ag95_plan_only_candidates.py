#!/usr/bin/env python3
"""Project DemoFunGrasp strategy into conservative AG95 pregrasp candidates."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("near-zero vector")
    return v / n


def quat_to_matrix(q):
    x, y, z, w = normalize(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def matrix_to_quat(r):
    t = float(np.trace(r))
    if t > 0:
        s = math.sqrt(t+1)*2
        q = [(r[2,1]-r[1,2])/s, (r[0,2]-r[2,0])/s, (r[1,0]-r[0,1])/s, .25*s]
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = math.sqrt(1+r[0,0]-r[1,1]-r[2,2])*2
            q = [.25*s, (r[0,1]+r[1,0])/s, (r[0,2]+r[2,0])/s, (r[2,1]-r[1,2])/s]
        elif i == 1:
            s = math.sqrt(1+r[1,1]-r[0,0]-r[2,2])*2
            q = [(r[0,1]+r[1,0])/s, .25*s, (r[1,2]+r[2,1])/s, (r[0,2]-r[2,0])/s]
        else:
            s = math.sqrt(1+r[2,2]-r[0,0]-r[1,1])*2
            q = [(r[0,2]+r[2,0])/s, (r[1,2]+r[2,1])/s, .25*s, (r[1,0]-r[0,1])/s]
    return normalize(q)


def signed_angle_about_z(a, b):
    a = normalize([a[0], a[1], 0])
    b = normalize([b[0], b[1], 0])
    return math.atan2(np.cross(a, b)[2], np.dot(a, b))


def rotate_z(v, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c*v[0]-s*v[1], s*v[0]+c*v[1], v[2]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", type=Path, required=True)
    ap.add_argument("--affordance", type=Path, required=True)
    ap.add_argument("--strategy", type=Path, required=True)
    ap.add_argument("--edited-reference", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--pregrasp-heights", default="0.14,0.12,0.10")
    ap.add_argument("--yaw-residuals-deg", default="0,-4,4,-8,8")
    ap.add_argument("--max-policy-yaw-deg", type=float, default=10.0)
    ap.add_argument("--max-final-yaw-deg", type=float, default=10.0)
    ap.add_argument("--snap-to-geometric-center", action="store_true",
                    help="Use the analytic top-center XY for two-finger execution while retaining the learned affordance score.")
    args = ap.parse_args()

    reg = yaml.safe_load(args.registration.read_text(encoding="utf-8"))
    aff = json.loads(args.affordance.read_text(encoding="utf-8"))
    strategy = json.loads(args.strategy.read_text(encoding="utf-8"))
    reference = np.load(args.edited_reference)
    grasp_i = int(strategy["grasp_reference_index"])
    wrist_r = quat_to_matrix(reference["wrist_quaternion_xyzw"][grasp_i])

    can_axis = normalize(reg["object_axis_base"])
    can_axis[2] = 0
    can_axis = normalize(can_axis)
    nominal_jaw = normalize(np.cross([0, 0, 1], can_axis))
    policy_jaw = wrist_r[:, 0].copy()  # Inspire local X used as thumb-index analogue.
    if np.linalg.norm(policy_jaw[:2]) < 1e-5:
        policy_jaw = nominal_jaw
    if np.dot(policy_jaw[:2], nominal_jaw[:2]) < 0:
        policy_jaw = -policy_jaw
    raw_policy_yaw = signed_angle_about_z(nominal_jaw, policy_jaw)
    max_yaw = math.radians(args.max_policy_yaw_deg)
    policy_yaw = float(np.clip(raw_policy_yaw, -max_yaw, max_yaw))
    policy_approach = normalize(wrist_r[:, 2])
    downward_alignment = float(np.dot(policy_approach, [0, 0, -1]))

    heights = [float(v) for v in args.pregrasp_heights.split(",")]
    residuals = [math.radians(float(v)) for v in args.yaw_residuals_deg.split(",")]
    learned_selected = np.asarray(aff["selected_point_base_m"], dtype=np.float64)
    selected = learned_selected.copy()
    if args.snap_to_geometric_center:
        object_center = np.asarray(reg["object_center_base_m"], dtype=np.float64)
        base_t_object = np.asarray(reg["base_T_object"]["matrix_4x4"], dtype=np.float64)
        top_direction = normalize(base_t_object[:3, 0])
        diameter = float(reg["dimensions_m"]["diameter"])
        top_center = object_center + 0.5 * diameter * top_direction
        selected[:2] = top_center[:2]
        selected[2] = top_center[2]
    candidates = []
    for height in heights:
        for residual in residuals:
            yaw_unclamped = policy_yaw + residual
            yaw = float(np.clip(yaw_unclamped,
                                -math.radians(args.max_final_yaw_deg),
                                math.radians(args.max_final_yaw_deg)))
            y_axis = normalize(rotate_z(nominal_jaw, yaw))
            x_axis = np.array([0.0, 0.0, -1.0])
            z_axis = normalize(np.cross(x_axis, y_axis))
            y_axis = normalize(np.cross(z_axis, x_axis))
            q = matrix_to_quat(np.column_stack((x_axis, y_axis, z_axis)))
            score = float(aff["selected_scores"]["fusion"]
                          - 0.45*abs(residual) - 0.10*abs(yaw) - 0.30*abs(height-0.12))
            candidates.append({
                "grasp_id": f"dfg_ag95_pregrasp_h{int(round(height*1000)):03d}_r{math.degrees(residual):+03.0f}",
                "score": score,
                "frame_id": "base_link",
                "position": {"x": float(selected[0]), "y": float(selected[1]), "z": float(selected[2]+height)},
                "orientation": {"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])},
                "metadata": {
                    "kind": "DEMOfunGrasp_guided_AG95_PLAN_ONLY_pregrasp",
                    "target_link": "gripper_center_link",
                    "local_x_axis": "vertical_down_approach",
                    "local_y_axis": "AG95_jaw_opening_projected_from_DFG_local_X",
                    "pregrasp_height_above_affordance_m": height,
                    "policy_yaw_clamped_rad": policy_yaw,
                    "yaw_residual_rad": residual,
                    "final_yaw_before_clamp_rad": yaw_unclamped,
                    "final_yaw_rad": yaw,
                    "no_execution": True,
                },
            })
    candidates.sort(key=lambda c: c["score"], reverse=True)

    n = np.asarray(reg["table_plane_base_abcd"][:3], dtype=np.float64)
    d = float(reg["table_plane_base_abcd"][3])
    center = np.asarray(reg["object_center_base_m"], dtype=np.float64)
    table_z = float(-(n[0]*center[0] + n[1]*center[1] + d) / n[2])
    saturation = int(np.sum(np.abs(strategy["policy_action_13d"]) > 0.95))
    output = {
        "schema": "demofungrasp_ag95_plan_only_candidates_v1",
        "safety": "PLAN_ONLY_NO_EXECUTION",
        "object_id": "001_chips_can",
        "frame_id": "base_link",
        "object_center_base_m": center.tolist(),
        "object_axis_base": can_axis.tolist(),
        "selected_affordance_base_m": learned_selected.tolist(),
        "execution_anchor_base_m": selected.tolist(),
        "execution_anchor_mode": "analytic_top_center" if args.snap_to_geometric_center else "learned_affordance_point",
        "execution_anchor_offset_from_affordance_m": (selected-learned_selected).tolist(),
        "selected_affordance_fusion_score": aff["selected_scores"]["fusion"],
        "strategy_projection": {
            "mapping_assumption": "Inspire wrist local X -> AG95 jaw axis; Inspire local Z -> approach axis",
            "raw_policy_yaw_rad": raw_policy_yaw,
            "clamped_policy_yaw_rad": policy_yaw,
            "policy_approach_downward_alignment": downward_alignment,
            "action_components_abs_gt_0_95": saturation,
            "constraint_note": "Policy direction is clamped; target, top-down approach, jaw geometry, and clearance are hardware constraints.",
        },
        "planning_scene": {
            "table_size_m": [0.80, 0.80, 0.04],
            "table_pose_xyz_m": [float(center[0]), float(center[1]), table_z-0.02],
            "object_size_m": [0.25, 0.075, 0.075],
        },
        "source_registration": str(args.registration),
        "source_affordance": str(args.affordance),
        "source_strategy": str(args.strategy),
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "candidate_count": len(candidates),
        "raw_policy_yaw_deg": math.degrees(raw_policy_yaw),
        "clamped_policy_yaw_deg": math.degrees(policy_yaw),
        "policy_approach_downward_alignment": downward_alignment,
        "table_height_base_m": table_z, "action_saturation_count": saturation,
        "top_candidate": candidates[0],
    }, indent=2))


if __name__ == "__main__":
    main()
