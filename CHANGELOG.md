# Changelog - GLM-4.7-NVFP4 Dynamo Deployment

## 2026-02-04 - Multi-Node Scaling and Fixes

### Added
- **4P+2D Configuration**: Scaled to 6 nodes (4 prefill @ TP=16, 2 decode @ TP=8)
- **AIPerf Benchmark Scripts**: `benchmark_glm47_aiperf.sh` and `quick_test_aiperf.sh`
- **Expert Parallelism Support**: `scripts/glm47_ep.sh` and `5_deploy_8node_ep4.sh` (requires newer container)
- **MNNVL Settings**: Multi-Node NVLink environment variables for faster cross-node communication

### Fixed
- **MoE Padding Assertion**: Patched `modelopt_quant.py` to allow padding with gated activations at TP=16
- **NCCL Timeout During CUDA Graph Capture**: Added extended timeouts and async error handling for multi-node decode workers
- **MoE Execution Stability**: Switched to `--moe-runner-backend flashinfer_trtllm` for more stable multi-node MoE inference

### Changed
- **Default Configuration**: Updated `4_use_dynamo_tools.sh` from 1P+1D to 4P+2D
- **Decode Worker Settings**: Added MNNVL flags (MC_FORCE_MNNVL, NCCL_MNNVL_ENABLE, NCCL_CUMEM_ENABLE)
- **NCCL Configuration**: Increased timeout to 3600s for long-running CUDA graph operations

### Configuration Details

#### Environment Variables Added (Decode Workers)
```bash
MC_FORCE_MNNVL=1              # Force Multi-Node NVLink
NCCL_MNNVL_ENABLE=1           # Enable MNNVL in NCCL
NCCL_CUMEM_ENABLE=1           # Enable CUDA unified memory
NCCL_TIMEOUT=3600             # 1-hour timeout for graph capture
NCCL_ASYNC_ERROR_HANDLING=1   # Better error handling
FLASHINFER_WORKSPACE_BASE=/fsw-home
TORCH_EXTENSIONS_DIR=/fsw-home/.cache/torch_extensions
TRITON_CACHE_DIR=/fsw-home/.triton/cache
```

#### Command-Line Flags Added
```bash
--moe-runner-backend flashinfer_trtllm  # Optimized MoE backend
--moe-dense-tp-size 1                   # MoE dense layer TP configuration
```

## 2026-02-03 - Initial Setup

### Added
- Single-node deployment (`1_single_node.sbatch`)
- Official Dynamo disaggregated deployment (`4_use_dynamo_tools.sh`)
- Custom GLM-4.7 worker script (`scripts/glm47.sh`)
- Kubernetes deployment manifests (single, 1P+1D, 1P+2D)
- Initial ModelOpt patches for weight scale checks

### Performance
- **2-node (1P+1D)**: ~2,100 tokens/sec with concurrency testing
