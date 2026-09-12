#!/bin/bash
PYTHON_ENV=/home/bkabadayi/anaconda3/etc/profile.d/conda.sh
source ${PYTHON_ENV}

source /etc/profile.d/modules.sh
module load cuda/11.7

echo "Running Physhead (data-sample)"
conda activate physhead-public

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
pwd

nvidia-smi

CFG=condor_bald/RHL466_bald_sample.yaml
MODEL_PATH=$(python3 -c "from omegaconf import OmegaConf; print(OmegaConf.load('${CFG}').model_path)")

echo "Model Path: ${MODEL_PATH}"

python train_bald.py --cfg ${CFG}

echo "Training completed, now testing:"
python render.py -m ${MODEL_PATH} --skip_train --skip_val --select_camera_id 0 --white_background

echo "Physhead (data-sample) finished"
