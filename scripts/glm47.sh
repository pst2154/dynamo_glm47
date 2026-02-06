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

# Install Dynamo if not present
if ! python3 -c "import dynamo" 2>/dev/null; then
    echo "=== Installing NVIDIA Dynamo ==="
    pip install --quiet nvidia-dynamo 2>/dev/null || echo "Warning: Could not install dynamo via pip"
fi

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

# Use TP_SIZE from env if set; otherwise TOTAL_GPUS. EP_SIZE from env (default 1).
TP_SIZE=${TP_SIZE:-$TOTAL_GPUS}
EP_SIZE=${EP_SIZE:-1}
echo "Using TP=$TP_SIZE EP=$EP_SIZE"

# Apply GLM-4.7 patch for modelopt quantization (DISABLED - using stock SGLang)
# echo "=== Applying GLM-4.7 patch ==="
# QUANT_FILE=$(python3 -c "import sglang,os;print(os.path.join(os.path.dirname(sglang.__file__),'srt/layers/quantization/modelopt_quant.py'))")
# if [ -f "/configs/modelopt_quant.py" ]; then
#     cp /configs/modelopt_quant.py "$QUANT_FILE"
#     echo "Patch applied"
# else
#     echo "Warning: Patch file not found, continuing without patch"
# fi

if [ "$mode" = "prefill" ]; then
    set -x
    export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800

    command_suffix=""
    if [[ -n "${DUMP_CONFIG_PATH}" ]]; then command_suffix="${command_suffix} --dump-config-to ${DUMP_CONFIG_PATH}"; fi

    # base_fp4: expert_parallel_size only, no DeepEP; moe_runner_backend flashinfer_trtllm
    ep_args=""
    if [ "$EP_SIZE" -gt 1 ]; then
        ep_args="--ep-size $EP_SIZE"
    fi

    PYTHONUNBUFFERED=1 \
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
    DYN_DECODE_KV_TRANSFER_TIMEOUT=10000 \
    MC_FORCE_MNNVL=1 \
    NCCL_MNNVL_ENABLE=1 \
    NCCL_CUMEM_ENABLE=1 \
    UCX_LOG_LEVEL=info \
    UCX_TLS=rc,cuda \
    UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1 \
    FLASHINFER_WORKSPACE_BASE=/lustre/fsw/portfolios/general/users/asteiner/.cache \
    TORCH_EXTENSIONS_DIR=/lustre/fsw/portfolios/general/users/asteiner/.cache/torch_extensions \
    TRITON_CACHE_DIR=/lustre/fsw/portfolios/general/users/asteiner/.cache/triton \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    SGLANG_ENABLE_JIT_DEEPGEMM=false \
    SGLANG_ENABLE_FLASHINFER_FP8_GEMM=1 \
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
    DYN_SKIP_SGLANG_LOG_FORMATTING=1 \
    SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN=1 \
    SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2=1 \
    DYN_METRICS_REQUEST_DURATION_COUNT=50 \
    DYN_METRICS_TTFT_COUNT=50 \
    DYN_METRICS_ITL_COUNT=50 \
    NIXL_DEBUG=INFO \
    python3 -m dynamo.sglang \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend nixl \
        --served-model-name glm-4.7 \
        --model-path /model/ \
        --trust-remote-code \
        --kv-cache-dtype fp8_e4m3 \
        --attention-backend trtllm_mha \
        --quantization modelopt_fp4 \
        --fp4-gemm-backend flashinfer_trtllm \
        --moe-runner-backend flashinfer_trtllm \
        --chunked-prefill-size 8192 \
        --stream-interval 10 \
        --watchdog-timeout 1000000 \
        --mem-fraction-static 0.80 \
        --max-running-requests 512 \
        --cuda-graph-max-bs 512 \
        --enable-symm-mem \
        --enable-nan-detection \
        --moe-dense-tp-size 1 \
        --scheduler-recv-interval 1 \
        --data-parallel-size 2 \
        --pipeline-parallel-size 1 \
        --enable-dp-attention \
        --load-balance-method round_robin \
        --disaggregation-bootstrap-port 30001 \
        --tensor-parallel-size "$TP_SIZE" \
        $ep_args \
        --dist-init-addr "$HOST_IP_MACHINE:$PORT" \
        --nnodes "$TOTAL_NODES" \
        --node-rank "$RANK" \
        --host 0.0.0.0 ${command_suffix}

elif [ "$mode" = "decode" ]; then
    set -x
    export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800

    command_suffix=""
    if [[ -n "${DUMP_CONFIG_PATH}" ]]; then command_suffix="${command_suffix} --dump-config-to ${DUMP_CONFIG_PATH}"; fi

    # base_fp4: expert_parallel_size only, no DeepEP; moe_runner_backend flashinfer_trtllm
    ep_args=""
    if [ "$EP_SIZE" -gt 1 ]; then
        ep_args="--ep-size $EP_SIZE"
    fi

    PYTHONUNBUFFERED=1 \
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
    SGLANG_DECODE_BOOTSTRAP_TIMEOUT=1000 \
    SGLANG_MOE_NVFP4_DISPATCH=1 \
    NCCL_TIMEOUT=3600 \
    NCCL_ASYNC_ERROR_HANDLING=1 \
    MC_FORCE_MNNVL=1 \
    NCCL_MNNVL_ENABLE=1 \
    NCCL_CUMEM_ENABLE=1 \
    FLASHINFER_WORKSPACE_BASE=/lustre/fsw/portfolios/general/users/asteiner/.cache \
    TORCH_EXTENSIONS_DIR=/lustre/fsw/portfolios/general/users/asteiner/.cache/torch_extensions \
    TRITON_CACHE_DIR=/lustre/fsw/portfolios/general/users/asteiner/.cache/triton \
    DYN_DECODE_KV_TRANSFER_TIMEOUT=10000 \
    UCX_LOG_LEVEL=info \
    UCX_TLS=rc,cuda \
    UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1 \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    SGLANG_ENABLE_JIT_DEEPGEMM=false \
    SGLANG_ENABLE_FLASHINFER_FP8_GEMM=1 \
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
    DYN_SKIP_SGLANG_LOG_FORMATTING=1 \
    SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN=1 \
    SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2=1 \
    DYN_METRICS_REQUEST_DURATION_COUNT=50 \
    DYN_METRICS_TTFT_COUNT=50 \
    DYN_METRICS_ITL_COUNT=50 \
    NCCL_DEBUG=INFO \
    NIXL_DEBUG=INFO \
    python3 -m dynamo.sglang \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend nixl \
        --served-model-name glm-4.7 \
        --model-path /model/ \
        --prefill-round-robin-balance \
        --trust-remote-code \
        --kv-cache-dtype fp8_e4m3 \
        --attention-backend trtllm_mha \
        --quantization modelopt_fp4 \
        --fp4-gemm-backend flashinfer_trtllm \
        --moe-runner-backend flashinfer_trtllm \
        --disaggregation-bootstrap-port 30001 \
        --stream-interval 10 \
        --watchdog-timeout 1000000 \
        --mem-fraction-static 0.90 \
        --max-running-requests 512 \
        --cuda-graph-max-bs 512 \
        --load-balance-method total_tokens \
        --load-watch-interval 2 \
        --enable-symm-mem \
        --enable-nan-detection \
        --moe-dense-tp-size 1 \
        --scheduler-recv-interval 1 \
        --data-parallel-size 1 \
        --tensor-parallel-size "$TP_SIZE" \
        $ep_args \
        --dist-init-addr "$HOST_IP_MACHINE:$PORT" \
        --nnodes "$TOTAL_NODES" \
        --node-rank "$RANK" \
        --host 0.0.0.0 ${command_suffix}
fi
