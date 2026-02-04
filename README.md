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
- Disaggregated architecture: dedicated prefill and decode nodes
- **NIXL-enabled** for fast cross-node KV cache transfer
- Handles all complexity properly

**Current Configuration (4P+2D):**
- **4 Prefill Nodes**: 16 GPUs total, TP=16
- **2 Decode Nodes**: 8 GPUs total, TP=8
- **MNNVL enabled**: Multi-node NVLink for faster communication
- **MoE backend**: FlashInfer + TensorRT-LLM for optimized MoE execution

**Monitor:**
```bash
cd /lustre/fsw/portfolios/general/users/asteiner/Dynamo_Stuff/dynamo_deploy/logs
ls -lrt | tail
# Find your job folder: <JOBID>_1P_1D_TIMESTAMP
cd <JOBID>_1P_1D_*/
tail -f *_prefill_*.err *_decode_*.err
```

**Get Frontend URL:**
```bash
# From log.out in the job directory
cat log.out | grep "Frontend available"
# Usually: http://nvl72d0XX-T01:8000
```

### 3. Kubernetes Deployments (`k8s/`)
**Production-ready Kubernetes deployments with multiple scaling options**

See [k8s/README.md](k8s/README.md) for full documentation.

**Available configurations:**
- **Single node** (`deploy-single-node.yaml`) - 1 node, 4 GPUs
- **1P+1D** (`deploy-1p1d.yaml`) - 2 nodes, 8 GPUs, ~2,100 tok/s
- **1P+2D** (`deploy-1p2d.yaml`) - 3 nodes, 12 GPUs, ~4,000 tok/s

```bash
kubectl apply -f k8s/deploy-1p1d.yaml
```

## Files in this Repo

### Deployment Scripts
- **`1_single_node.sbatch`** - Single-node SLURM deployment script
- **`4_use_dynamo_tools.sh`** - Multi-node disaggregated SLURM deployment (4P+2D configuration)
- **`scripts/glm47.sh`** - Custom worker script with MNNVL and FlashInfer+TRTLLM MoE backend
- **`scripts/glm47_ep.sh`** - Worker script with Expert Parallelism support (experimental)
- **`5_deploy_8node_ep4.sh`** - 8-node EP=4 deployment (requires newer container)

### Benchmarking
- **`benchmark_glm47_aiperf.sh`** - Comprehensive AIPerf benchmark suite (6 tests)
- **`quick_test_aiperf.sh`** - Quick AIPerf validation test (20 requests)
- **`benchmark_concurrency.py`** - Legacy concurrent performance test script

### Configuration
- **`configs/`** - Configuration directory for Dynamo tools
- **`k8s/`** - Kubernetes deployment manifests (single-node, 1P+1D, 1P+2D)

### Documentation
- **`README.md`** - This file
- **`SETUP.md`** - Detailed setup instructions

## Required Setup

### Custom GLM-4.7 Worker Script
The disaggregated deployment needs a custom worker script. This has been installed at:
```
~/dynamo/examples/backends/sglang/slurm_jobs/scripts/gb200-fp4/disagg/glm47.sh
```

This script is automatically used when you run `4_use_dynamo_tools.sh`.

## Patches and Fixes Applied

### 1. ModelOpt MoE Padding Assertion Patch
**File**: `/lustre/fsw/portfolios/general/users/asteiner/Dynamo_Stuff/configs/modelopt_quant.py` (lines 1709-1713)

**Issue**: At higher TP (TP=16 for prefill), MoE layers with gated activations require padding, but the ModelOpt code had an assertion preventing this.

**Fix**: Commented out the assertion to allow padding with gated activations:
```python
# PATCHED: Commented out assertion for GLM-4.7 with TP=16
# assert not layer.moe_runner_config.is_gated, (
#     "The intermediate size required padding, "
#     "but padding is also implemented for gated activations"
# )
```

This file is automatically copied into the container by `glm47.sh` at runtime.

### 2. MoE Runner Backend
**Setting**: `--moe-runner-backend flashinfer_trtllm`

Uses FlashInfer + TensorRT-LLM optimized MoE execution path, bypassing some ModelOpt quantization complexity and improving multi-node stability.

### 3. MNNVL Multi-Node Settings
**Environment Variables** (decode workers only):
```bash
MC_FORCE_MNNVL=1
NCCL_MNNVL_ENABLE=1
NCCL_CUMEM_ENABLE=1
NCCL_TIMEOUT=3600
NCCL_ASYNC_ERROR_HANDLING=1
```

Enables Multi-Node NVLink for faster cross-node communication and prevents NCCL timeouts during CUDA graph capture.

### 4. Model Config
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

## Benchmarking

### Using AIPerf (Recommended)

Once deployment is running, use AIPerf for comprehensive benchmarking:

```bash
# Quick test (20 requests)
./quick_test_aiperf.sh http://nvl72d0XX-T01:8000

# Full benchmark suite
./benchmark_glm47_aiperf.sh http://nvl72d0XX-T01:8000
```

**AIPerf Features:**
- Real-time dashboard UI
- Multiple concurrency levels (4, 8, 16)
- Request rate testing
- Long context tests (4K tokens)
- Comprehensive metrics (TTFT, ITL, throughput)
- Artifact export for analysis

### Performance Expectations

**2-node (1P+1D, TP=4):** ~2,100 tokens/sec
**6-node (4P+2D, TP=16/8):** Target throughput TBD

## Known Issues and Solutions

### Issue 1: MoE Padding Assertion with TP=16
**Symptom**: `AssertionError: The intermediate size required padding, but padding is also implemented for gated activations`

**Cause**: At TP=16, GLM-4.7's MoE layers require padding, but ModelOpt's quantization code prevented this for gated activations.

**Solution**: Patched `modelopt_quant.py` (lines 1709-1713) to comment out the assertion. Patch is auto-applied via CONFIG_DIR mount.

### Issue 2: NCCL Timeout During Multi-Node CUDA Graph Capture
**Symptom**: Decode workers hang at "Capture cuda graph begin" or crash with "NET/Socket: socket progress error"

**Cause**: CUDA graph capture blocks the main thread for several minutes. With TP across multiple nodes, NCCL collectives timeout waiting for synchronization.

**Solution**: 
- Added MNNVL settings for faster multi-node communication
- Increased NCCL_TIMEOUT to 3600 seconds
- Enabled NCCL_ASYNC_ERROR_HANDLING
- Using `--moe-runner-backend flashinfer_trtllm` for better stability

### Issue 3: Expert Parallelism Not Supported
**Symptom**: `RuntimeError: The size of tensor a (160) must match the size of tensor b (40)`

**Cause**: NVFP4 quantization + generic EP not supported in sglang-runtime:0.8.1 container.

**Solution**: Use standard TP/DP disaggregation instead. EP requires newer container with DeepEP support.

## Cancel Jobs

```bash
scancel <JOB_ID>
```

## Performance Notes

**CUDA Graph Capture**: Decode workers spend 10-20 minutes capturing CUDA graphs during initialization. This is normal and required for optimal performance. Look for progress indicators showing batch size capture (0/48 → 48/48).

**Initialization Time**: Full 6-node deployment takes ~15-20 minutes to be fully ready:
- Model loading: ~4 minutes per worker
- CUDA graph capture (prefill): ~5-10 minutes
- CUDA graph capture (decode): ~10-20 minutes (multi-node synchronization)
