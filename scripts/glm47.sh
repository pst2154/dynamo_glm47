#!/bin/bash
# GLM-4.7-NVFP4 disaggregated setup

print_usage() {
    echo "Usage: $0 <mode>"
    echo "  mode: prefill or decode"
    exit 1
}

if [ $# -ne 1 ]; then
    echo "Error: Expected 1 argument, got $#"
    print_usage
fi

mode=$1

if [ "$mode" != "prefill" ] && [ "$mode" != "decode" ]; then
    echo "Error: mode must be 'prefill' or 'decode', got '$mode'"
    print_usage
fi

echo "Mode: $mode"
echo "Model: GLM-4.7-NVFP4"

# Check required environment variables
if [ -z "$HOST_IP_MACHINE" ]; then
    echo "Error: HOST_IP_MACHINE environment variable is not set"
    exit 1
fi

if [ -z "$PORT" ]; then
    echo "Error: PORT environment variable is not set"
    exit 1
fi

if [ -z "$TOTAL_GPUS" ]; then
    echo "Error: TOTAL_GPUS environment variable is not set"
    exit 1
fi

if [ -z "$RANK" ]; then
    echo "Error: RANK environment variable is not set"
    exit 1
fi

if [ -z "$TOTAL_NODES" ]; then
    echo "Error: TOTAL_NODES environment variable is not set"
    exit 1
fi

# Apply GLM-4.7 patch for modelopt quantization
echo "=== Applying GLM-4.7 patch ==="
QUANT_FILE=$(python3 -c "import sglang,os;print(os.path.join(os.path.dirname(sglang.__file__),'srt/layers/quantization/modelopt_quant.py'))")
if [ -f "/configs/modelopt_quant.py" ]; then
    cp /configs/modelopt_quant.py "$QUANT_FILE"
    echo "Patch applied"
else
    echo "Warning: Patch file not found, continuing without patch"
fi

if [ "$mode" = "prefill" ]; then
    set -x
    export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800

    command_suffix=""
    if [[ -n "${DUMP_CONFIG_PATH}" ]]; then command_suffix="${command_suffix} --dump-config-to ${DUMP_CONFIG_PATH}"; fi

    PYTHONUNBUFFERED=1 \
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
    MC_FORCE_MNNVL=1 \
    NCCL_MNNVL_ENABLE=1 \
    NCCL_CUMEM_ENABLE=1 \
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
    SGLANG_ENABLE_JIT_DEEPGEMM=false \
    SGLANG_ENABLE_FLASHINFER_GEMM=true \
    python3 -m dynamo.sglang \
        --disaggregation-mode prefill \
        --served-model-name glm-4.7 \
        --model-path /model/ \
        --trust-remote-code \
        --kv-cache-dtype fp8_e4m3 \
        --attention-backend flashinfer \
        --quantization modelopt_fp4 \
        --stream-interval 10 \
        --watchdog-timeout 600 \
        --mem-fraction-static 0.85 \
        --load-balance-method round_robin \
        --disaggregation-bootstrap-port 30001 \
        --enable-symm-mem \
        --tensor-parallel-size "$TOTAL_GPUS" \
        --dist-init-addr "$HOST_IP_MACHINE:$PORT" \
        --nnodes "$TOTAL_NODES" \
        --node-rank "$RANK" \
        --host 0.0.0.0 ${command_suffix}

elif [ "$mode" = "decode" ]; then
    set -x
    export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800

    command_suffix=""
    if [[ -n "${DUMP_CONFIG_PATH}" ]]; then command_suffix="${command_suffix} --dump-config-to ${DUMP_CONFIG_PATH}"; fi

    PYTHONUNBUFFERED=1 \
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
    SGLANG_DECODE_BOOTSTRAP_TIMEOUT=1000 \
    MC_FORCE_MNNVL=1 \
    NCCL_MNNVL_ENABLE=1 \
    NCCL_CUMEM_ENABLE=1 \
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
    SGLANG_ENABLE_JIT_DEEPGEMM=false \
    SGLANG_ENABLE_FLASHINFER_GEMM=true \
    python3 -m dynamo.sglang \
        --disaggregation-mode decode \
        --served-model-name glm-4.7 \
        --model-path /model/ \
        --prefill-round-robin-balance \
        --trust-remote-code \
        --kv-cache-dtype fp8_e4m3 \
        --attention-backend flashinfer \
        --quantization modelopt_fp4 \
        --disaggregation-bootstrap-port 30001 \
        --stream-interval 10 \
        --watchdog-timeout 600 \
        --mem-fraction-static 0.85 \
        --enable-symm-mem \
        --tensor-parallel-size "$TOTAL_GPUS" \
        --dist-init-addr "$HOST_IP_MACHINE:$PORT" \
        --nnodes "$TOTAL_NODES" \
        --node-rank "$RANK" \
        --host 0.0.0.0 ${command_suffix}
fi
