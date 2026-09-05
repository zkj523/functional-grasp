# Functional Grasp Deployment: Simulation Policy to UR5e Hardware

A single-view, model-based pipeline that takes a dexterous-hand functional grasping
policy trained in simulation and executes it on a **parallel-jaw gripper** — with a
pre-execution quality gate that makes perception failures and execution failures
separately attributable, and a per-trial archive that makes every result reproducible.


---

## Why this exists

A functional grasp has to contact the task-relevant region — the handle of a pair of
scissors, not its blades. Simulation policies for this problem are typically trained
on multi-fingered dexterous hands, whose compliance absorbs a lot of geometric
uncertainty. Most deployed arms do not have such a hand. Transferring the *behaviour*
is therefore not enough: the representation has to survive the change of embodiment,
and the execution stack has to be honest about when it should refuse to act.

Two design decisions shape everything here.

**Transfer the representation, not the trajectory.** The simulated Inspire hand's
joint trajectory is discarded. What crosses the boundary is the affordance point and
the approach direction; the gripper-specific quantities are recomputed on the real
side from the registered mesh.

**Refuse to execute on bad perception.** Every run passes a quality gate before any
motion command is issued. A run that fails the gate is recorded as a *perception*
failure and never becomes a misleading *grasp* failure.

---

## Pipeline

```
 single synchronized RGB-D capture          src/single_rgbd_ee_capture.py
   ↓
 table plane fit (RANSAC) + object segmentation
 analytic pose estimation for the observable DoF
                                            src/prepare_chips_can_registration.py
   ↓
 resample to 512x6 (xyz + normals), align to the simulator's scale convention
 affordance scoring                         src/predict_chips_can_affordance.py
   ↓
 one-step policy inference, demonstration edited offline
                                            src/run_demofungrasp_real_adapter.py
   ↓
 gripper pose candidates for the AG95        src/generate_ag95_plan_only_candidates.py
   ↓
 ┌─ QUALITY GATE ─────────────────────────┐  src/validate_online_chips_can_run.py
 │ registration p95 / mean residual        │
 │ table-plane fit RMSE                    │  fails ⇒ abort, log as perception failure
 │ visible object point count               │
 │ observed axis span vs nominal            │
 │ approach-direction alignment             │
 └────────────────────────────────────────┘
   ↓ (pass only)
 guarded Cartesian execution on UR5e + AG95  src/demo_chips_can_visual_servo_grasp.py
 align → rotate → descend → staged close → lift
   ↓
 freeze the run: manifest, checksums, evidence
                                            src/finalize_online_chips_can_run.py
```

`src/run_online_chips_can_grasp.sh` orchestrates the stages; exactly one mode runs
per invocation so that a partially-completed run can never be mistaken for a
complete one.

---

## What is worth reading first

| File | Lines | Why |
|---|---:|---|
| `src/prepare_chips_can_registration.py` | 310 | Analytic pose estimation. Thirteen small single-purpose functions: quaternion conversion, mask extraction, unprojection, RANSAC plane fit, normal estimation, farthest-point sampling, PLY export. The comments state plainly which DoF is **unobservable** — rotation about a cylinder's long axis is fixed by the table normal, not recovered from texture. |
| `src/validate_online_chips_can_run.py` | 86 | The quality gate. Small on purpose: it is a predicate over recorded numbers, with no side effects. |
| `src/demo_chips_can_visual_servo_grasp.py` | 782 | Execution node. Handles controller timeouts, TCP offsets, and the fact that the gripper reports the same state code when fully open as when holding nothing. |
| `src/finalize_online_chips_can_run.py` | 288 | Archiving. Freezes a run into manifest + code snapshot + SHA-256 checksums + post-execution evidence. |

---

## Reproducibility

Every trial is frozen into a self-describing record:

```
real_grasp_<timestamp>_<object>_<position>_<trial>_<result>/
├── manifest.yaml        object, pose, method, checkpoints, result, timestamps
├── quality_gate.json    every gate metric and its threshold
├── registration/        estimated pose, residual distances, resampled cloud
├── capture/             the RGB-D frame the decision was made from
├── evidence/            post-execution images
├── logs/                full terminal output
├── code_snapshot/       the exact scripts that ran
└── CHECKSUMS.sha256     integrity of all of the above
```

An archived run can be re-read years later without reference to the machine that
produced it. `examples/` contains one sanitized record.

---

## Measured perception quality

YCB chips can, five table positions, two repetitions each (mean ± s.d.):

| Metric | Value |
|---|---|
| registration p95 distance | 8.98 ± 1.12 mm |
| registration mean distance | 5.80 ± 0.59 mm |
| table plane fit RMSE | 1.57 ± 0.06 mm |
| visible object points | 74112 ± 4384 |
| observed long-axis span | 216.3 ± 1.1 mm (nominal 250 mm) |
| end-effector high-pose error | 0.08 ± 0.04 mm |

The 216.3 mm against a 250 mm nominal length is not an error — it is self-occlusion
of the far end under single-view observation, and it is one of the gated quantities.

**These are perception-stage numbers. They do not stand in for grasp success rate**,
which is being evaluated systematically and is not reported here.

---

## Hardware and software

UR5e · DH Robotics AG95 · Intel RealSense D435i (eye-in-hand) · ROS Noetic · MoveIt ·
Python 3 · NumPy · OpenCV

All paths are environment-driven — set `FG_UR_WS`, `FG_SIM_ROOT`, `FG_RUN_ROOT`,
`FG_ARCHIVE_ROOT`, `FG_ROS_PYTHON`, `FG_DEMO_PYTHON` as needed.

## Relationship to prior work

The simulation-side policy is DemoFunGrasp (Mao et al., *Universal Dexterous
Functional Grasping via Demonstration-Editing Reinforcement Learning*,
arXiv:2512.13380), used unmodified. This repository is the hardware deployment built
on top of it and the adaptation from a five-fingered hand to a parallel-jaw gripper.
No real-world data is used in training, so the policy transfer is zero-shot.

## License

MIT — see `LICENSE`.
