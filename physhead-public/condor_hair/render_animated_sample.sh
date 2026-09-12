#!/bin/bash
# Renders the animated (physics-simulated) hair motion for the data-sample p50 checkpoint,
# driven by a target pose-transfer sequence. Run interactively (not a condor job).
#
# NOTE: --target_path (-t) MUST be an absolute path. Unlike --source_path, target_path is
# never passed through os.path.abspath() anywhere in the codebase, and
# readCamerasFromTransforms() double-joins it in a way that silently produces broken,
# duplicated paths when it's relative (verified — do not change this to a repo-relative path).

source /home/bkabadayi/anaconda3/etc/profile.d/conda.sh
conda activate physhead-public

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH=experiments/20230313--1653--RHL466_hair_p50_sample
TARGET_PATH="${REPO_ROOT}/data-sample/RHL466/pose_transfer/rightleft"
CFG=cfgs/RHL466_p50_sample.yml
ITERATION=30000
SELECT_CAMERA_ID=8

python render_hair_animation.py -m ${MODEL_PATH} -t ${TARGET_PATH} \
    --select_camera_id ${SELECT_CAMERA_ID} --iteration ${ITERATION} --cfg ${CFG}
