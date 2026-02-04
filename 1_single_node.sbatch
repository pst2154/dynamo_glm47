#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=8:00:00
#SBATCH -A general_cs_infra
#SBATCH -p batch_long
#SBATCH --job-name=dynamo-glm47-single
#SBATCH --output=/lustre/fsw/portfolios/general/users/asteiner/dynamo_glm47/logs/dynamo_single_%j.log
#SBATCH --error=/lustre/fsw/portfolios/general/users/asteiner/dynamo_glm47/logs/dynamo_single_%j.log

set -e

echo "=========================================="
echo "=== Dynamo + SGLang Single Node Setup ==="
echo "=========================================="
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="
echo ""

# Configuration
BASE_DIR=/lustre/fsw/portfolios/general/users/asteiner
MODEL_DIR=${BASE_DIR}/GLM-4.7-NVFP4
MODEL_NAME=glm-4.7-nvfp4
FRONTEND_PORT=8000
WORKER_PORT=30000

# Create logs directory
mkdir -p ${BASE_DIR}/Dynamo_Stuff/logs

echo "=== Configuration ==="
echo "Model: $MODEL_DIR"
echo "Container: nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.8.1"
echo "Frontend Port: $FRONTEND_PORT"
echo "Worker Port: $WORKER_PORT"
echo "Tensor Parallel: 4"
echo ""

# Run in container with Dynamo + SGLang
srun --container-image=docker://nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.8.1 \
     --container-mounts=${BASE_DIR}:${BASE_DIR} \
     bash -c '
set -e

export HF_HOME=/lustre/fsw/portfolios/general/users/asteiner/.cache/huggingface
export TRANSFORMERS_CACHE=/lustre/fsw/portfolios/general/users/asteiner/.cache/huggingface
export VLLM_CACHE_ROOT=/tmp/vllm_cache
export XDG_CACHE_HOME=/tmp/cache
export PIP_CACHE_DIR=/tmp/pip_cache

cd /tmp

echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

echo "=== Installing patched SGLang ==="
QUANT_FILE=$(python3 -c "import sglang,os;print(os.path.join(os.path.dirname(sglang.__file__),\"srt/layers/quantization/modelopt_quant.py\"))")
cp /lustre/fsw/portfolios/general/users/asteiner/dynamo_glm47/patches/modelopt_quant.py "$QUANT_FILE"
echo "Patched file installed"
echo ""

echo "=== Starting Dynamo Frontend ==="
python3 -m dynamo.frontend --http-port 8000 --store-kv file &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"
sleep 5

echo ""
echo "=== Starting SGLang Worker ==="
PYTHONUNBUFFERED=1 \
python3 -m dynamo.sglang \
  --model-path /lustre/fsw/portfolios/general/users/asteiner/GLM-4.7-NVFP4 \
  --served-model-name glm-4.7 \
  --store-kv file \
  --tp 4 \
  --port 30000 \
  --trust-remote-code \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend flashinfer \
  --mem-fraction-static 0.85 \
  --watchdog-timeout 600
'

echo ""
echo "=== Job completed at: $(date) ==="
