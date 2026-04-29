#!/usr/bin/env bash
# Run the full CS 136 preprocessing pipeline on a configurable subset of ACDC.
#
# Defaults: 20 random frames per weather condition from the train split (80 total).
# Override TRAIN_DIR / PER_CONDITION / SPLIT via environment variables, e.g.:
#
#   TRAIN_DIR=/path/to/rgb_anon PER_CONDITION=50 ./run_all.sh
#
# Bails on the first failing step. Times every step so you can see which
# algorithm dominates runtime.

set -euo pipefail

TRAIN_DIR="${TRAIN_DIR:-/Users/jenilkathrotiya/Downloads/rgb_anon_trainvaltest/rgb_anon}"
PER_CONDITION="${PER_CONDITION:-20}"
SPLIT="${SPLIT:-train}"

cd "$(dirname "$0")/.."  # SafeMask/

ARGS=(--input-dir "$TRAIN_DIR" --split "$SPLIT" --per-condition "$PER_CONDITION")

STEPS=(
  "Part1.1 Gaussian        | Part1_Preprocessing/01_Gaussian_Filter/gaussian_filter.py"
  "Part1.2 Sobel           | Part1_Preprocessing/02_Sobel_Edge/sobel_edge.py"
  "Part1.3 Canny           | Part1_Preprocessing/03_Canny_Edge/canny_edge.py"
  "Part1.4 Hough           | Part1_Preprocessing/04_Hough_Transform/hough_transform.py"
  "Part1.5 Edge eval       | Part1_Preprocessing/05_Edge_Detector_Evaluation/evaluate_edges.py"
  "Part1.6 Texture seg     | Part1_Preprocessing/06_Texture_Segmentation/texture_segmentation.py"
  "Part2.1 CLAHE+Bilateral | Part2_Creative/01_CLAHE_Bilateral_Canny/clahe_bilateral_canny.py"
  "Part2.2 Morph cleanup   | Part2_Creative/02_Morphological_Cleanup/morph_cleanup.py"
  "Part2.3 Lab mean-shift  | Part2_Creative/03_Color_Texture_Lab/color_texture_lab.py"
  "Part3.1 Distortions     | Part3_Robustness/01_Distortions/apply_distortions.py"
  "Part3.2 Compare         | Part3_Robustness/02_Pipeline_Comparison/compare_pipelines.py"
)

echo ">>> Pipeline start"
echo ">>> input  : $TRAIN_DIR"
echo ">>> split  : $SPLIT"
echo ">>> sample : $PER_CONDITION per condition"
echo

global_start=$(date +%s)
for entry in "${STEPS[@]}"; do
  label="${entry%% |*}"
  script="${entry##*| }"
  script="${script# }"
  echo "================================================================"
  echo ">>> $label  ($script)"
  echo "================================================================"
  step_start=$(date +%s)
  python3 "cs136_preprocessing/$script" "${ARGS[@]}"
  step_end=$(date +%s)
  printf ">>> %s done in %ds\n\n" "$label" "$((step_end - step_start))"
done
global_end=$(date +%s)

echo "================================================================"
echo ">>> Pipeline finished in $((global_end - global_start))s"
echo "================================================================"
