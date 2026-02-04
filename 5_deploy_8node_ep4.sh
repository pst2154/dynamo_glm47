#!/bin/bash
# 8-node deployment with EP=4 for both prefill and decode
# 4 prefill nodes + 4 decode nodes = 8 nodes total (32 GPUs)

set -e

BASE_DIR=/lustre/fsw/portfolios/general/users/asteiner
DYNAMO_DIR=${BASE_DIR}/dynamo/examples/backends/sglang/slurm_jobs
WORK_DIR=${BASE_DIR}/dynamo_glm47/dynamo_deploy
REPO_DIR=${BASE_DIR}/dynamo_glm47

# Create work directory
mkdir -p ${WORK_DIR}
cd ${WORK_DIR}

# Copy Dynamo tools
cp ${DYNAMO_DIR}/submit_job_script.py .
cp ${DYNAMO_DIR}/job_script_template_disagg.j2 .
cp -r ${DYNAMO_DIR}/scripts .

# Override with our EP-enabled script
cp ${REPO_DIR}/scripts/glm47_ep.sh scripts/gb200-fp4/disagg/glm47_ep.sh
chmod +x scripts/gb200-fp4/disagg/glm47_ep.sh

mkdir -p logs outputs

# Configuration
MODEL_DIR=${BASE_DIR}/GLM-4.7-NVFP4
CONFIG_DIR=${REPO_DIR}/configs
CONTAINER_IMAGE="docker://nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.8.1"
GPUS_PER_NODE=4
NETWORK_INTERFACE="eth0"
ACCOUNT="general_cs_infra"
PARTITION="batch_long"
TIME_LIMIT="4:00:00"

# 8-node setup: 4 prefill + 4 decode with EP=4
PREFILL_NODES=4
DECODE_NODES=4
PREFILL_WORKERS=1
DECODE_WORKERS=1
EP_SIZE=4

echo "=============================================="
echo "===   8-Node EP=4 Deployment (GLM-4.7)   ===="
echo "=============================================="
echo "Model: GLM-4.7-NVFP4"
echo "Prefill: ${PREFILL_NODES} nodes, EP=${EP_SIZE}"
echo "Decode:  ${DECODE_NODES} nodes, EP=${EP_SIZE}"
echo "Total:   $((PREFILL_NODES + DECODE_NODES)) nodes, $((PREFILL_NODES * GPUS_PER_NODE + DECODE_NODES * GPUS_PER_NODE)) GPUs"
echo "Expert Parallelism: ${EP_SIZE}"
echo "Tensor Parallelism: ${GPUS_PER_NODE} (per node)"
echo "=============================================="
echo ""

# Set EP_SIZE environment variable for the worker script
export EP_SIZE=${EP_SIZE}

python3 submit_job_script.py \
  --model-dir ${MODEL_DIR} \
  --config-dir ${CONFIG_DIR} \
  --container-image ${CONTAINER_IMAGE} \
  --gpus-per-node ${GPUS_PER_NODE} \
  --network-interface ${NETWORK_INTERFACE} \
  --prefill-nodes ${PREFILL_NODES} \
  --decode-nodes ${DECODE_NODES} \
  --prefill-workers ${PREFILL_WORKERS} \
  --decode-workers ${DECODE_WORKERS} \
  --account ${ACCOUNT} \
  --partition ${PARTITION} \
  --time-limit ${TIME_LIMIT} \
  --gpu-type gb200-fp4 \
  --script-variant glm47_ep \
  --disable-config-dump \
  --enable-multiple-frontends \
  --num-additional-frontends 0 \
  --extra-slurm-args "gres=gpu:4"

echo ""
echo "✅ Job submitted!"
echo ""
echo "Logs: ${WORK_DIR}/logs/"
echo ""
echo "Monitor with:"
echo "  cd ${WORK_DIR}/logs"
echo "  ls -lrt | tail"
echo "  # Find your job folder: <JOBID>_4P_4D_*"
echo "  tail -f <JOBID>_4P_4D_*/log.err"
echo ""
