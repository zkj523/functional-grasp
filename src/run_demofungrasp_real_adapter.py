#!/usr/bin/env python3
"""Run the trained one-step policy and edit its Inspire demonstration offline.

This produces a strategy-space wrist reference. It is deliberately not a UR5e
trajectory: the Inspire wrist frame still has to be mapped to the AG95 tool frame
and checked by MoveIt before any execution is considered.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

import isaacgym  # noqa: F401 -- Isaac Gym must be imported before torch.
import torch


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = np.moveaxis(np.asarray(q1), -1, 0)
    x2, y2, z2, w2 = np.moveaxis(np.asarray(q2), -1, 0)
    return np.stack((
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ), axis=-1)


def quat_from_euler_xyz(roll, pitch, yaw):
    sr, cr = np.sin(roll/2), np.cos(roll/2)
    sp, cp = np.sin(pitch/2), np.cos(pitch/2)
    sy, cy = np.sin(yaw/2), np.cos(yaw/2)
    return np.array((
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
        cr*cp*cy + sr*sp*sy,
    ), dtype=np.float64)


def quat_rotate(q, points):
    q = np.asarray(q, dtype=np.float64)
    qvec = q[:3]
    uv = np.cross(np.broadcast_to(qvec, np.asarray(points).shape), points)
    uuv = np.cross(np.broadcast_to(qvec, np.asarray(points).shape), uv)
    return np.asarray(points) + 2.0 * (q[3] * uv + uuv)


def matrix_to_quat_xyzw(r):
    # Stable eigenvector form; sign is normalized for reproducible output.
    k = np.array([
        [r[0,0]-r[1,1]-r[2,2], r[1,0]+r[0,1], r[2,0]+r[0,2], r[1,2]-r[2,1]],
        [r[1,0]+r[0,1], r[1,1]-r[0,0]-r[2,2], r[2,1]+r[1,2], r[2,0]-r[0,2]],
        [r[2,0]+r[0,2], r[2,1]+r[1,2], r[2,2]-r[0,0]-r[1,1], r[0,1]-r[1,0]],
        [r[1,2]-r[2,1], r[2,0]-r[0,2], r[0,1]-r[1,0], r.trace()],
    ]) / 3.0
    values, vectors = np.linalg.eigh(k)
    q = vectors[:, np.argmax(values)]
    if q[3] < 0:
        q = -q
    return q


def load_policy(repo, config_path, checkpoint, device):
    # ManiSkill's utility package calls exit(0) during import when NVML has no
    # driver, even for CPU inference. Its GPU telemetry is unused here.
    if device.type == "cpu":
        try:
            import pynvml
            pynvml.nvmlInit = lambda: None
        except ImportError:
            pass
    sys.path.insert(0, str(repo))
    from algo.ppo_onestep import ActorCritic

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    model_cfg = dict(cfg["train"]["params"]["policy"])
    # The environment appends xyz only (512*3) after 21 state values.
    model_cfg["pc_shape"] = [512, 3]
    model = ActorCritic(
        obs_shape=(7+7+3+4+512*3,), states_shape=(0,), actions_shape=(13,),
        initial_std=cfg["train"]["params"]["init_noise_std"],
        model_cfg=model_cfg, asymmetric=False, use_pcl=True,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    model.eval()
    return model, cfg


def edit_reference(reference, action, style_row, cfg):
    wrist = np.asarray(reference["wrist_initobj_pos"], dtype=np.float64).copy()
    quats = np.asarray(reference["wrist_quat"], dtype=np.float64).copy()
    hand = np.asarray(reference["hand_qpos"], dtype=np.float64).copy()

    xyz_range = np.asarray(cfg["task"]["env"]["randomizeTrackingReferenceRange"][:3])
    ang_range = np.asarray(cfg["task"]["env"]["randomizeTrackingReferenceRange"][3:6])
    delta = action[:3] * xyz_range
    random_quat = quat_from_euler_xyz(*(action[3:6] * ang_range))
    wrist = quat_rotate(random_quat, wrist) + delta
    quats = quat_multiply(np.broadcast_to(random_quat, quats.shape), quats)

    lift_t = int(cfg["task"]["env"]["trackingReferenceLiftTimestep"])
    wrist[lift_t:] = (np.asarray(reference["wrist_initobj_pos"])[lift_t:]
                      - np.asarray(reference["wrist_initobj_pos"])[lift_t-1]
                      + wrist[lift_t-1])

    func = cfg["task"]["func"]
    q_lo, q_hi = func["qpos_delta_scale"]
    q_delta = q_lo + 0.5 * (action[6:12] + 1.0) * (q_hi - q_lo)
    scale_lo, scale_hi = func["scale_limit"]
    q_scale = scale_lo + 0.5 * (action[12] + 1.0) * (scale_hi - scale_lo)
    target = style_row[:6] * q_scale + q_delta
    initial = hand[0].copy()
    denominator = hand[lift_t-1] - initial
    fraction = (target - initial) / (denominator + 1e-6)
    hand[:lift_t-1] = initial + (hand[:lift_t-1] - initial) * fraction
    hand[lift_t-1:] = target
    return wrist, quats, hand, delta, random_quat, q_delta, q_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--affordance", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--style-dict", type=Path, required=True)
    parser.add_argument("--style", type=int, default=0, choices=range(4))
    parser.add_argument("--nominal-object-origin", nargs=3, type=float, default=(0.50, -0.10, 0.10))
    parser.add_argument("--nominal-eef-pose", nargs=7, type=float,
                        default=(0.50, 0.00, 0.35, 0.0, 0.0, 0.0, 1.0))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    reg = yaml.safe_load(args.registration.read_text(encoding="utf-8"))
    base_t_object = np.asarray(reg["base_T_object"]["matrix_4x4"], dtype=np.float64)
    object_q = matrix_to_quat_xyzw(base_t_object[:3, :3])
    object_origin_policy = np.asarray(args.nominal_object_origin, dtype=np.float64)
    object_pose_policy = np.r_[object_origin_policy, object_q]

    points = np.load(args.points).astype(np.float64)
    if points.shape != (512, 6):
        raise ValueError(f"Expected 512x6 canonical points, got {points.shape}")
    xyz_policy = (base_t_object[:3, :3] @ points[:, :3].T).T + object_origin_policy
    affordance = json.loads(args.affordance.read_text(encoding="utf-8"))
    affordance_object = np.asarray(affordance["selected_point_object_m"], dtype=np.float64)
    affordance_policy = base_t_object[:3, :3] @ affordance_object + object_origin_policy
    style_onehot = np.eye(4, dtype=np.float32)[args.style]

    # Source-code order: eefpose, objinitpose, affordance, style, then flattened objpcl xyz.
    observation = np.concatenate((np.asarray(args.nominal_eef_pose), object_pose_policy,
                                  affordance_policy, style_onehot, xyz_policy.reshape(-1))).astype(np.float32)
    if observation.shape != (1557,):
        raise AssertionError(observation.shape)

    device = torch.device(args.device)
    policy, cfg = load_policy(args.repo, args.config, args.checkpoint, device)
    with torch.no_grad():
        action = policy(torch.from_numpy(observation).unsqueeze(0).to(device), inference=True)[0].cpu().numpy()

    with args.reference.open("rb") as f:
        reference = pickle.load(f)
    styles = np.load(args.style_dict)
    wrist_rel, wrist_q, hand_q, xyz_delta, random_q, q_delta, q_scale = edit_reference(
        reference, action, styles[args.style], cfg)
    wrist_base = wrist_rel + base_t_object[:3, 3]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "policy_observation_1557.npy", observation)
    np.savez(args.output_dir / "edited_inspire_reference.npz",
             wrist_object_relative_m=wrist_rel.astype(np.float32),
             wrist_base_m=wrist_base.astype(np.float32),
             wrist_quaternion_xyzw=wrist_q.astype(np.float32),
             inspire_hand_qpos=hand_q.astype(np.float32))
    grasp_index = int(cfg["task"]["env"]["trackingReferenceLiftTimestep"]) - 1
    selected_base = np.asarray(affordance["selected_point_base_m"])
    result = {
        "schema": "demofungrasp_real_strategy_adapter_v1",
        "safety": "OFFLINE_STRATEGY_ONLY_NOT_A_UR5E_TRAJECTORY",
        "observation_order": ["eefpose_7", "objinitpose_7", "affordance_3", "style_onehot_4", "objpcl_xyz_1536"],
        "style": args.style,
        "style_semantics": "style 0 uses thumb+index contacts and is the closest Inspire analogue to a two-finger gripper" if args.style == 0 else "Inspire hand style; no direct AG95 joint mapping",
        "policy_action_13d": action.astype(float).tolist(),
        "decoded_wrist_translation_m": xyz_delta.astype(float).tolist(),
        "decoded_wrist_left_rotation_quaternion_xyzw": random_q.astype(float).tolist(),
        "decoded_inspire_qpos_delta": q_delta.astype(float).tolist(),
        "decoded_inspire_qpos_scale": float(q_scale),
        "policy_object_pose_xyzw": object_pose_policy.astype(float).tolist(),
        "policy_eef_pose_xyzw": list(map(float, args.nominal_eef_pose)),
        "grasp_reference_index": grasp_index,
        "inspire_wrist_grasp_pose_base_xyzw": np.r_[wrist_base[grasp_index], wrist_q[grasp_index]].astype(float).tolist(),
        "inspire_wrist_to_selected_affordance_distance_m": float(np.linalg.norm(wrist_base[grasp_index] - selected_base)),
        "checkpoint": str(args.checkpoint),
        "registration": str(args.registration),
        "next_required_mapping": "Inspire wrist strategy -> collision-checked AG95 tool0 grasp pose",
    }
    (args.output_dir / "strategy_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
