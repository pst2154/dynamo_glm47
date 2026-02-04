# Dynamo + SGLang GLM-4.7-NVFP4 Deployment

## Setup Summary

**Model**: GLM-4.7-NVFP4 (MoE with 160 experts, 92 layers)
**Quantization**: NVFP4 with FP8 KV cache
**Hardware**: GB200 NVL72 (4 GPUs per node)
**Container**: `nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.8.1`

## Deployment Options

### 1. Single Node (`1_single_node.sbatch`) ✅
**Simple single-node deployment - WORKS**

```bash
cd /lustre/fsw/portfolios/general/users/asteiner/dynamo_glm47
sbatch 1_single_node.sbatch
```

**Architecture:**
- 1 node with 4 GPUs
- Dynamo frontend + SGLang worker on same node
- File-based KV store (no etcd/NATS needed)
- TP=4 (Tensor Parallelism across 4 GPUs)

**Access:**
```bash
# Check job and get node
squeue -u $USER
# Assuming job is on nvl72d012-T09
curl http://nvl72d012-T09:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "glm-4.7", "prompt": "Hello!", "max_tokens": 50}'
```

### 2. Official Dynamo Disaggregated (`4_use_dynamo_tools.sh`) ✅
**Multi-node disaggregated prefill/decode using Dynamo's official tools - RECOMMENDED**

```bash
cd /lustre/fsw/portfolios/general/users/asteiner/dynamo_glm47
bash 4_use_dynamo_tools.sh
```

**What it does:**
- Uses Dynamo's `submit_job_script.py` with custom GLM-4.7 config
- Proper multi-node coordination (etcd, NATS, frontend, nginx)
- Disaggregated architecture: prefill node + decode node
- **NIXL-enabled** for fast cross-node KV cache transfer
- Handles all complexity properly

**Architecture:**
- **Services node**: etcd + NATS + frontend
- **Prefill node**: Prompt processing (TP=4)
- **Decode node**: Token generation (TP=4)

**Monitor:**
```bash
cd ~/dynamo/examples/backends/sglang/slurm_jobs/logs
ls -lrt | tail
# Find your job folder: 65970_1P_1D_TIMESTAMP
cd 65970_1P_1D_*/
tail -f *_prefill_*.err *_decode_*.err
```

## Files in this Repo

- **`1_single_node.sbatch`** - Single-node deployment script
- **`4_use_dynamo_tools.sh`** - Multi-node disaggregated deployment using Dynamo's official tools
- **`patches/modelopt_quant.py`** - Required patch for GLM-4.7-NVFP4 quantization
- **`configs/`** - Configuration directory for Dynamo tools
- **`logs/`** - Job output logs
- **`README.md`** - This file

## Required Setup

### Custom GLM-4.7 Worker Script
The disaggregated deployment needs a custom worker script. This has been installed at:
```
~/dynamo/examples/backends/sglang/slurm_jobs/scripts/gb200-fp4/disagg/glm47.sh
```

This script is automatically used when you run `4_use_dynamo_tools.sh`.

## Patches Applied

### ModelOpt Quantization Patch
**File**: `patches/modelopt_quant.py`

Disables two assertion checks in SGLang for GLM-4.7-NVFP4:
1. Weight scale dimension divisibility check
2. Weight scale dtype check

Copied into container at runtime.

### Model Config
**File**: `GLM-4.7-NVFP4/generation_config.json`

Added missing `bos_token_id: 151329` to enable model registration.

## Logs

**Single Node:**
- Logs: `~/dynamo_glm47/logs/dynamo_single_<JOB_ID>.log`

**Official Dynamo Disaggregated:**
- Logs: `~/dynamo_glm47/dynamo_deploy/logs/<JOB_ID>_1P_1D_*/`
- Contains: `log.out`, `log.err`, per-node worker logs, GPU utilization

## Key Parameters

### Common
- `--served-model-name glm-4.7` - Model alias (use instead of full path)
- `--quantization modelopt_fp4` - FP4 quantization
- `--kv-cache-dtype fp8_e4m3` - FP8 KV cache
- `--attention-backend flashinfer` - FlashInfer attention (GLM-specific)
- `--trust-remote-code` - Required for GLM-4.7
- `--mem-fraction-static 0.85` - GPU memory allocation
- `--enable-symm-mem` - Symmetric memory for better performance

### Parallelism
- `--tp <N>` - Tensor Parallelism (within node)
- `--dp <N>` - Data Parallelism (decode only)
- `--ep-size <N>` - Expert Parallelism (across nodes)

### Disaggregation
- `--disaggregation-mode {prefill,decode}` - Worker role
- `--disaggregation-bootstrap-port 30001` - Coordination port
- `--prefill-round-robin-balance` - Decode load balancing

### KV Transfer Backend
- `--disaggregation-transfer-backend nixl` - NVIDIA NIXL for efficient cross-node KV cache transfer
  - Auto-selects optimal transport (NVLink, InfiniBand, etc.)
  - GPU-to-GPU direct memory transfers when available
  - Falls back to standard network when needed

## Troubleshooting

### Job stuck or workers not starting
- Check node resource contention: `squeue -u $USER`
- Verify logs: `tail -f logs/*<JOB_ID>*`
- Ensure nodes aren't busy with other jobs

### Model loading errors
- Verify model path: `/lustre/fsw/portfolios/general/users/asteiner/GLM-4.7-NVFP4`
- Check patch is applied in logs
- Ensure `bos_token_id` in `generation_config.json`

### Connection timeouts
- Wait 2-3 minutes for model loading (large MoE model)
- Check worker logs for "Model loaded" or similar
- Verify frontend is running and port 8000 is accessible

## Cancel Jobs

```bash
scancel <JOB_ID>
```
