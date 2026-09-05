#!/usr/bin/env python3
"""Freeze one online chips-can run as a paper-ready experiment record."""

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


# Paths are environment-driven so the pipeline runs on any machine.
# Override with FG_RUN_ROOT / FG_ARCHIVE_ROOT / FG_UR_WS / FG_SIM_ROOT.
_ENV = os.environ.get
ONLINE_ROOT = Path(_ENV("FG_RUN_ROOT", "~/demo/runs/online_chips_can")).expanduser().resolve()
ARCHIVE_ROOT = Path(_ENV("FG_ARCHIVE_ROOT", "~/demo/runs")).expanduser().resolve()
UR_WS = Path(_ENV("FG_UR_WS", "~/ur_ws")).expanduser()
RAL = Path(_ENV("FG_SIM_ROOT", "~/RAL-simulate")).expanduser()

CODE_FILES = [
    UR_WS / "demo/scripts/run_online_chips_can_grasp.sh",
    UR_WS / "demo/scripts/validate_online_chips_can_run.py",
    UR_WS / "demo/scripts/finalize_online_chips_can_run.py",
    UR_WS / "src/ur_controller/scripts/single_rgbd_ee_capture.py",
    UR_WS / "src/ur_controller/scripts/demo_grasp_plan_preview.py",
    UR_WS / "src/ur_controller/scripts/demo_chips_can_semiauto_grasp.py",
    UR_WS / "src/ur_controller/launch/demo_grasp_plan_preview_dfg_ag95.launch",
    UR_WS / "src/ur_controller/launch/demo_chips_can_full_grasp_guarded.launch",
    RAL / "real_robot_bridge/prepare_chips_can_registration.py",
    RAL / "real_robot_bridge/predict_chips_can_affordance.py",
    RAL / "real_robot_bridge/run_demofungrasp_real_adapter.py",
    RAL / "real_robot_bridge/generate_ag95_plan_only_candidates.py",
    RAL / "real_robot_bridge/visualize_chips_can_bridge.py",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--result", required=True, choices=("success", "failure"))
    parser.add_argument("--note", default="")
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_online_run(path):
    resolved = path.resolve()
    if resolved == ONLINE_ROOT or ONLINE_ROOT not in resolved.parents:
        raise RuntimeError("run-dir must be a child of %s" % ONLINE_ROOT)
    if not resolved.is_dir():
        raise RuntimeError("run-dir does not exist: %s" % resolved)
    return resolved


def git_head(repo):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_execution_log(path):
    metrics = {}
    if not path.is_file():
        return metrics
    text = path.read_text(encoding="utf-8", errors="replace")
    fractions = [float(v) for v in re.findall(r"Cartesian fraction:\s*([0-9.]+)", text)]
    descent_steps = re.findall(r"descend to grasp height step\s+(\d+)", text)
    lift_steps = re.findall(r"lift test step\s+(\d+)", text)
    high_errors = re.findall(r"High-pose final position error:\s*([0-9.]+)\s*m", text)
    scalings = re.findall(
        r"High-pregrasp speed scaling: velocity=([0-9.]+) acceleration=([0-9.]+)",
        text,
    )
    finished = "Semi-auto grasp sequence finished." in text
    if fractions:
        metrics["cartesian_fraction_min"] = min(fractions)
        metrics["cartesian_segment_count"] = len(fractions)
    if descent_steps:
        metrics["descent_steps"] = max(int(v) for v in descent_steps)
    if lift_steps:
        metrics["lift_steps"] = max(int(v) for v in lift_steps)
    if high_errors:
        metrics["high_pose_final_position_error_m"] = float(high_errors[-1])
    if scalings:
        metrics["pregrasp_velocity_scaling"] = float(scalings[-1][0])
        metrics["pregrasp_acceleration_scaling"] = float(scalings[-1][1])
    metrics["sequence_finished_cleanly"] = finished
    return metrics


def write_checksums(root):
    checksum_path = root / "CHECKSUMS.sha256"
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p != checksum_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append("%s  %s" % (digest, path.relative_to(root)))
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def verify_checksums(root):
    for line in (root / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError("checksum mismatch: %s" % relative)


def main():
    args = parse_args()
    source = ensure_online_run(Path(args.run_dir))

    quality_path = source / "quality_gate.json"
    registration_path = source / "registration/registration.yaml"
    candidate_path = source / "registration/ag95_plan_only_candidates_base.json"
    required = (quality_path, registration_path, candidate_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("cannot finalize; missing: %s" % ", ".join(missing))

    quality = load_json(quality_path)
    if args.result == "success" and not quality.get("passed", False):
        raise RuntimeError("refusing success archive because quality gate did not pass")

    evidence = sorted((source / "evidence").glob("*_metadata.json"))
    if args.result == "success" and not evidence:
        raise RuntimeError("refusing success archive without after-execution evidence")

    archive_name = "real_grasp_%s_%s" % (source.name, args.result)
    destination = ARCHIVE_ROOT / archive_name
    if destination.exists():
        raise RuntimeError(
            "archive already exists: %s; use a fresh capture_prepare run for each formal trial"
            % destination
        )

    temp = ARCHIVE_ROOT / (".%s.tmp-%d" % (archive_name, os.getpid()))
    try:
        shutil.copytree(source, temp)
        snapshot = temp / "code_snapshot"
        snapshot.mkdir()
        for source_file in CODE_FILES:
            if not source_file.is_file():
                raise RuntimeError("missing code snapshot file: %s" % source_file)
            shutil.copy2(source_file, snapshot / source_file.name)

        registration = load_yaml(registration_path)
        candidate_data = load_json(candidate_path)
        candidate = candidate_data["candidates"][0]
        selected_path = source / "registration/affordance/selected_affordance.json"
        selected = load_json(selected_path) if selected_path.is_file() else {}
        latest_evidence = load_json(evidence[-1]) if evidence else None
        execution_logs = sorted((source / "logs").glob("execute_*.log"))
        if not execution_logs and (source / "logs/execute.log").is_file():
            execution_logs = [source / "logs/execute.log"]
        designated_execution_log = execution_logs[-1] if execution_logs else None
        execution_metrics = (
            parse_execution_log(designated_execution_log)
            if designated_execution_log is not None
            else {}
        )

        manifest = {
            "schema": "ral_real_grasp_auto_archive_v1",
            "run_id": archive_name,
            "source_online_run": str(source),
            "finalized_local": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "result": {
                "label": args.result,
                "user_declared": True,
                "note": args.note or None,
                "after_execution_evidence_count": len(evidence),
            },
            "object": {
                "id": "001_chips_can",
                "dataset": "YCB",
                "dimensions_m": [0.075, 0.075, 0.250],
                "pose": "side_lying",
            },
            "robot": {
                "arm": "UR5e",
                "gripper": "DH Robotics AG95",
                "camera": "Intel RealSense D435i",
                "camera_mount": "eye_in_hand",
            },
            "method": {
                "affordance": "0.75 geometric score + 0.25 contact-aware PointNet",
                "affordance_region": "strict_middle_20_percent_argmax",
                "selected_affordance_fusion_score": selected.get("selected_scores", {}).get("fusion"),
                "policy_checkpoint": str(RAL / "checkpoints/final_mixed/seed0/model_4060.pt"),
                "policy_style": 0,
                "adaptation": "DemoFunGrasp_Inspire_direction_to_AG95_top_down_cross_diameter",
                "execution_mode": "single_view_registration_then_guarded_open_loop",
            },
            "registration": {
                "rgb_depth_sync_delta_sec": registration.get("rgb_depth_sync_delta_sec"),
                "table_plane_rmse_m": registration.get("table_plane_rmse_m"),
                "visible_object_points": registration.get("object_visible_points"),
                "observed_axis_span_m": registration.get("observed_axis_span_m"),
                "observed_to_model_distance_m": registration.get("observed_to_model_distance_m"),
                "object_center_base_m": registration.get("object_center_base_m"),
                "object_axis_base": registration.get("object_axis_base"),
            },
            "quality_gate": quality,
            "execution": {
                "candidate_id": candidate.get("grasp_id"),
                "designated_log": (
                    str(designated_execution_log.relative_to(source))
                    if designated_execution_log is not None
                    else None
                ),
                "high_target_base_m": [
                    candidate["position"]["x"],
                    candidate["position"]["y"],
                    candidate["position"]["z"],
                ],
                **execution_metrics,
            },
            "designated_evidence": latest_evidence,
            "software": {
                "ros_distro": "noetic",
                "ur_ws_git_head": git_head(UR_WS),
                "ral_git_head": git_head(RAL),
                "code_snapshot": "code_snapshot/",
            },
        }
        (temp / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        readme = """# Automatically archived YCB chips-can real grasp

- Result: **{result}** (explicitly declared by the experimenter)
- Source run: `{source}`
- Quality gate passed: `{quality}`
- After-execution evidence samples: `{evidence_count}`
- Execution mode: single-view registration followed by guarded open-loop execution

The archive contains the complete online run, per-step terminal logs, an exact
code snapshot, a machine-readable `manifest.yaml`, and SHA-256 integrity hashes.
The latest evidence sample is designated in the manifest. A saved evidence image
alone is not treated as proof of success; the result label is the experimenter's
explicit observation supplied to the `finalize` command.
""".format(
            result=args.result,
            source=source,
            quality=quality.get("passed"),
            evidence_count=len(evidence),
        )
        (temp / "README.md").write_text(readme, encoding="utf-8")
        file_count = write_checksums(temp)
        verify_checksums(temp)
        temp.rename(destination)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise

    print("FINALIZED_ARCHIVE=%s" % destination)
    print("RESULT=%s" % args.result)
    print("HASHED_FILES=%d" % file_count)
    print("CHECKSUM_VERIFICATION=passed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FINALIZE_ERROR: %s" % exc, file=sys.stderr)
        sys.exit(2)
