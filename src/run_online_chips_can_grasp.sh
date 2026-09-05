#!/usr/bin/env bash
set -euo pipefail

# Reusable single-view chips-can pipeline. Exactly one mode runs per invocation.

# Paths are environment-driven; override any of these before invoking.
WS=${FG_UR_WS:-$HOME/ur_ws}
RAL=${FG_SIM_ROOT:-$HOME/RAL-simulate}
RUN_ROOT=${FG_RUN_ROOT:-$HOME/demo/runs/online_chips_can}
ROS_PYTHON=${FG_ROS_PYTHON:-/usr/bin/python3}
DEMO_PYTHON=${FG_DEMO_PYTHON:-$HOME/anaconda3/envs/demo/bin/python}
CANONICAL="$RAL/assets/real_object_dataset/pointclouds/001_chips_can.npy"
AFFORDANCE_CHECKPOINT="$RAL/checkpoints/affordance/best_model.pt"
POLICY_CHECKPOINT="$RAL/checkpoints/final_mixed/seed0/model_4060.pt"
POLICY_CONFIG="$RAL/checkpoints/final_mixed/seed0/config.json"
REFERENCE="$RAL/tasks/grasp_ref_inspire.pkl"
STYLE_DICT="$RAL/dataset_processor/inspire_static_style_cali.npy"

source "$WS/devel/setup.bash"

usage() {
  echo "Usage:"
  echo "  $0 capture_prepare [label]"
  echo "  $0 preview RUN_DIR"
  echo "  $0 execute RUN_DIR RUN_NEW_CHIPS_CAN_GRASP"
  echo "  $0 finalize RUN_DIR success|failure [note]"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 2
  fi
}

capture_prepare() {
  local label="${1:-trial}"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local run_dir="$RUN_ROOT/${stamp}_${label}"
  local capture_dir="$run_dir/capture"
  local registration_dir="$run_dir/registration"
  mkdir -p "$capture_dir" "$registration_dir" "$run_dir/logs"
  exec > >(tee -a "$run_dir/logs/capture_prepare.log") 2>&1

  require_file "$CANONICAL"
  require_file "$AFFORDANCE_CHECKPOINT"
  require_file "$POLICY_CHECKPOINT"
  require_file "$POLICY_CONFIG"
  require_file "$REFERENCE"
  require_file "$STYLE_DICT"

  echo "[1/7] Capturing synchronized RGB-D and capture-time TF"
  rosrun ur_controller single_rgbd_ee_capture.py \
    --out-dir "$capture_dir" --prefix chips_can \
    --base-frame base_link --ee-frame gripper_center_link \
    --camera-frame camera_color_optical_frame

  local metadata
  metadata="$(find "$capture_dir" -maxdepth 1 -name '*_metadata.json' -print -quit)"
  require_file "$metadata"

  echo "[2/7] Registering YCB 001_chips_can"
  "$ROS_PYTHON" "$RAL/real_robot_bridge/prepare_chips_can_registration.py" \
    --metadata "$metadata" --canonical "$CANONICAL" --output-dir "$registration_dir"

  echo "[3/7] Selecting strict middle-body fused affordance"
  "$DEMO_PYTHON" "$RAL/real_robot_bridge/predict_chips_can_affordance.py" \
    --registration "$registration_dir/registration.yaml" \
    --points "$registration_dir/affordance_input_512x6_object.npy" \
    --checkpoint "$AFFORDANCE_CHECKPOINT" \
    --output-dir "$registration_dir/affordance" \
    --alpha 0.75 --middle-fraction 0.20 --topk-ratio 0.20 \
    --selection-mode argmax --seed 7

  echo "[4/7] Running final mixed DemoFunGrasp strategy adapter"
  "$DEMO_PYTHON" "$RAL/real_robot_bridge/run_demofungrasp_real_adapter.py" \
    --repo "$RAL" \
    --registration "$registration_dir/registration.yaml" \
    --points "$registration_dir/affordance_input_512x6_object.npy" \
    --affordance "$registration_dir/affordance/selected_affordance.json" \
    --checkpoint "$POLICY_CHECKPOINT" --config "$POLICY_CONFIG" \
    --reference "$REFERENCE" --style-dict "$STYLE_DICT" \
    --style 0 --device cpu --output-dir "$registration_dir/demofungrasp_strategy"

  echo "[5/7] Projecting strategy to one AG95 high candidate"
  "$ROS_PYTHON" "$RAL/real_robot_bridge/generate_ag95_plan_only_candidates.py" \
    --registration "$registration_dir/registration.yaml" \
    --affordance "$registration_dir/affordance/selected_affordance.json" \
    --strategy "$registration_dir/demofungrasp_strategy/strategy_report.json" \
    --edited-reference "$registration_dir/demofungrasp_strategy/edited_inspire_reference.npz" \
    --output "$registration_dir/ag95_plan_only_candidates_base.json" \
    --pregrasp-heights 0.20 --yaw-residuals-deg 0 \
    --max-policy-yaw-deg 3 --max-final-yaw-deg 3 \
    --snap-to-geometric-center

  echo "[6/7] Applying registration and geometry quality gate"
  "$ROS_PYTHON" "$WS/demo/scripts/validate_online_chips_can_run.py" --run-dir "$run_dir"

  echo "[7/7] Generating review images"
  "$ROS_PYTHON" "$RAL/real_robot_bridge/visualize_chips_can_bridge.py" \
    --registration-dir "$registration_dir"

  mkdir -p "$RUN_ROOT"
  ln -sfn "$run_dir" "$RUN_ROOT/latest"
  echo
  echo "PREPARED_RUN_DIR=$run_dir"
  echo "Next: $0 preview $run_dir"
}

preview_run() {
  local run_dir="$1"
  local candidate="$run_dir/registration/ag95_plan_only_candidates_base.json"
  require_file "$run_dir/quality_gate.json"
  require_file "$candidate"
  mkdir -p "$run_dir/logs"
  local preview_log="$run_dir/logs/preview_$(date +%Y%m%d_%H%M%S).log"
  exec > >(tee "$preview_log") 2>&1
  "$ROS_PYTHON" "$WS/demo/scripts/validate_online_chips_can_run.py" --run-dir "$run_dir"
  roslaunch ur_controller demo_grasp_plan_preview_dfg_ag95.launch \
    candidate_file:="$candidate" candidate_index:=0
}

execute_run() {
  local run_dir="$1"
  local confirmation="${2:-}"
  if [[ "$confirmation" != "RUN_NEW_CHIPS_CAN_GRASP" ]]; then
    echo "Refusing execution: third argument must be RUN_NEW_CHIPS_CAN_GRASP" >&2
    exit 3
  fi
  local candidate="$run_dir/registration/ag95_plan_only_candidates_base.json"
  require_file "$candidate"
  mkdir -p "$run_dir/logs"
  local execute_log="$run_dir/logs/execute_$(date +%Y%m%d_%H%M%S).log"
  exec > >(tee "$execute_log") 2>&1
  "$ROS_PYTHON" "$WS/demo/scripts/validate_online_chips_can_run.py" --run-dir "$run_dir"
  mkdir -p "$run_dir/evidence"
  roslaunch ur_controller demo_chips_can_full_grasp_guarded.launch \
    candidate_file:="$candidate" \
    execute:=true confirm_execute:=RUN_CHIPS_CAN_SEMIAUTO
  rosrun ur_controller single_rgbd_ee_capture.py \
    --out-dir "$run_dir/evidence" --prefix after_lift \
    --base-frame base_link --ee-frame gripper_center_link \
    --camera-frame camera_color_optical_frame
  echo "Completed run: $run_dir"
}

finalize_run() {
  local run_dir="$1"
  local result="$2"
  local note="${3:-}"
  "$ROS_PYTHON" "$WS/demo/scripts/finalize_online_chips_can_run.py" \
    --run-dir "$run_dir" --result "$result" --note "$note"
}

mode="${1:-}"
case "$mode" in
  capture_prepare)
    capture_prepare "${2:-trial}"
    ;;
  preview)
    [[ $# -eq 2 ]] || { usage; exit 2; }
    preview_run "$2"
    ;;
  execute)
    [[ $# -eq 4 || $# -eq 3 ]] || { usage; exit 2; }
    execute_run "$2" "${3:-}"
    ;;
  finalize)
    [[ $# -ge 3 && $# -le 4 ]] || { usage; exit 2; }
    finalize_run "$2" "$3" "${4:-}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
