#!/bin/bash
# Use Dynamo's official deployment tools

set -e

BASE_DIR=/lustre/fsw/portfolios/general/users/asteiner
DYNAMO_DIR=${BASE_DIR}/dynamo/examples/backends/sglang/slurm_jobs
WORK_DIR=${BASE_DIR}/Dynamo_Stuff/dynamo_deploy

# Create temporary work directory for Dynamo tools
mkdir -p ${WORK_DIR}
cd ${WORK_DIR}

# Copy Dynamo tools to work directory
cp ${DYNAMO_DIR}/submit_job_script.py .
cp ${DYNAMO_DIR}/job_script_template_disagg.j2 .
cp -r ${DYNAMO_DIR}/scripts .
mkdir -p logs outputs

# Configuration
MODEL_DIR=${BASE_DIR}/GLM-4.7-NVFP4
CONFIG_DIR=${BASE_DIR}/Dynamo_Stuff/configs
CONTAINER_IMAGE="docker://nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.8.1"
GPUS_PER_NODE=4
NETWORK_INTERFACE="eth0"  # Common interface, or use: ib0, enp0s31f6, etc
ACCOUNT="general_cs_infra"
PARTITION="batch_long"
TIME_LIMIT="4:00:00"

# Disaggregated: 4 prefill workers + 2 decode workers (1 node each, TP=4)
PREFILL_NODES=4
DECODE_NODES=2
PREFILL_WORKERS=4
DECODE_WORKERS=2

echo "=============================================="
echo "=== Dynamo Official Disaggregated Deploy ===="
echo "=============================================="
echo "Model: GLM-4.7-NVFP4"
echo "Config: ${CONFIG_DIR}"
echo "Prefill: ${PREFILL_NODES} nodes, ${PREFILL_WORKERS} workers"
echo "Decode: ${DECODE_NODES} nodes, ${DECODE_WORKERS} workers"
echo "Total: $((PREFILL_NODES + DECODE_NODES)) nodes, $((PREFILL_NODES * GPUS_PER_NODE + DECODE_NODES * GPUS_PER_NODE)) GPUs"
echo "GPU Type: gb200-fp4"
echo "=============================================="
echo ""

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
  --script-variant glm47 \
  --disable-config-dump \
  --enable-multiple-frontends \
  --num-additional-frontends 0 \
  --extra-slurm-args "gres=gpu:4"

echo ""
echo "Job submitted!"
echo "Logs location: ${WORK_DIR}/logs/"
echo ""
echo "To monitor:"
echo "  cd ${WORK_DIR}/logs"
echo "  ls -lrt | tail"
echo "  tail -f *_1P_1D_*/log.err"
echo ""
echo "Symlink to Dynamo_Stuff/logs:"
echo "  ln -s ${WORK_DIR}/logs/* ${BASE_DIR}/Dynamo_Stuff/logs/"
