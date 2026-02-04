# GLM-4.7-NVFP4 Dynamo Setup - Working Configuration

## What's Included

This repo contains the **working configuration** for deploying GLM-4.7-NVFP4 (160-expert MoE, NVFP4 quantized) with NVIDIA Dynamo + SGLang on GB200 NVL72.

## Quick Start

### Option 1: Single Node (Simplest)
```bash
cd ~/dynamo_glm47
sbatch 1_single_node.sbatch
# Check: squeue -u $USER
# Logs: ~/dynamo_glm47/logs/dynamo_single_<JOB_ID>.log
```

### Option 2: Disaggregated Prefill/Decode (Production)
```bash
cd ~/dynamo_glm47
bash 4_use_dynamo_tools.sh
# Logs: ~/dynamo_glm47/dynamo_deploy/logs/<JOB_ID>_1P_1D_*/
```

## Current Status

**Job 65975 is RUNNING** - Disaggregated setup successfully deployed:
- **Nodes**: nvl72d012-T[05,09]
- **Prefill**: nvl72d012-T05 (loading model...)
- **Decode**: nvl72d012-T09 (loading model...)
- **Status**: Both workers loading model successfully (15-20% complete)

Monitor:
```bash
cd ~/dynamo_glm47/dynamo_deploy/logs/65975_1P_1D_20260203_203643/
tail -f nvl72d012-T05_prefill_w0.err nvl72d012-T09_decode_w0.err
```

## Architecture

### Single Node
- 1 node, 4 GPUs
- TP=4 (Tensor Parallelism)
- File-based KV store
- Frontend + Worker co-located

### Disaggregated
- **Node 0**: etcd + NATS + Frontend + Prefill worker (TP=4)
- **Node 1**: Decode worker (TP=4)
- Coordinated via etcd/NATS
- Prefill handles prompt processing
- Decode handles token generation

## Key Files

| File | Purpose |
|------|---------|
| `1_single_node.sbatch` | Single-node SLURM job |
| `4_use_dynamo_tools.sh` | Disaggregated deployment using Dynamo's official tools |
| `patches/modelopt_quant.py` | Patched SGLang file for GLM-4.7 NVFP4 support |
| `configs/modelopt_quant.py` | Patch file (copied into container at runtime) |
| `~/dynamo/examples/.../glm47.sh` | Custom worker script for Dynamo tools |

## Dependencies

- **Container**: `nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.8.1`
- **Model**: `/lustre/fsw/portfolios/general/users/asteiner/GLM-4.7-NVFP4`
- **Dynamo Tools**: `~/dynamo/examples/backends/sglang/slurm_jobs/`
- **Hardware**: GB200 NVL72 (4 GPUs/node, ~180GB memory/GPU)

## What Was Fixed

1. ✅ **ModelOpt Quantization**: Disabled FP4 weight scale assertions
2. ✅ **Model Config**: Added missing `bos_token_id: 151329`
3. ✅ **Network Interface**: Changed from `enP6p9s0np0` to `eth0`
4. ✅ **Dynamo Integration**: Created custom `glm47.sh` worker script
5. ✅ **Log Location**: Moved logs from Dynamo's default to `~/dynamo_glm47/`

## Testing

Once model finishes loading (~2-3 min total):

```bash
# Get the prefill node from job output
curl http://nvl72d012-T05:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Next Steps

- [ ] Test inference after model loads
- [ ] Add Expert Parallelism (EP) for wider distribution
- [ ] Benchmark throughput/latency
- [ ] Scale to more nodes
