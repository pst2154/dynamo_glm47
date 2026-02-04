# GLM-4.7-NVFP4 Kubernetes Deployments

Kubernetes deployment manifests for GLM-4.7-NVFP4 using NVIDIA Dynamo on GB200 clusters.

## Deployment Options

### 1. Single Node (`deploy-single-node.yaml`)
**Resources**: 1 node, 4 GPUs
- Simplest setup, no disaggregation
- TP=4 within single node
- Good for: Development, testing, low-throughput workloads

**Performance**: ~300-500 tok/s sustained

```bash
kubectl apply -f deploy-single-node.yaml
```

### 2. Disaggregated 1P+1D (`deploy-1p1d.yaml`)
**Resources**: 2 nodes, 8 GPUs (4 per node)
- 1 Prefill worker (TP=4)
- 1 Decode worker (TP=4)
- NIXL transfer backend for KV cache

**Performance**: ~2,100 tok/s at 10 concurrent requests

**Use when**:
- Need high throughput with concurrency
- Want specialized prefill/decode optimization
- Production workloads with moderate load

```bash
kubectl apply -f deploy-1p1d.yaml
```

### 3. Disaggregated 1P+2D (`deploy-1p2d.yaml`)
**Resources**: 3 nodes, 12 GPUs (4 per node)
- 1 Prefill worker (TP=4)
- 2 Decode workers (TP=4 each)
- 2x decode capacity for high concurrency

**Performance**: ~4,000+ tok/s expected (2x decode throughput)

**Use when**:
- Very high concurrency workloads (50+ concurrent)
- Decode is the bottleneck
- Maximum throughput required

```bash
kubectl apply -f deploy-1p2d.yaml
```

## Prerequisites

1. **Kubernetes cluster** with NVIDIA Dynamo operator installed
2. **Model files**: GLM-4.7-NVFP4 model stored in PVC at `/opt/model/GLM-4.7-NVFP4`
3. **GPU nodes**: GB200 NVL72 nodes with 4 GPUs each
4. **PVC**: Persistent volume claim named `model-cache` (pre-created)

## Configuration

### Model Path
Update the model path if different:
```yaml
args:
  - --model-path
  - /opt/model/GLM-4.7-NVFP4  # Change this
```

### Memory Settings
Adjust memory fraction if needed:
```yaml
args:
  - --mem-fraction-static
  - "0.85"  # 0.7-0.9 range, tune based on KV cache needs
```

### Timeout Settings
For very long generation tasks:
```yaml
args:
  - --watchdog-timeout
  - "1200"  # Increase from 600s
```

## Monitoring

### Check deployment status
```bash
kubectl get dynamographdeployment
kubectl describe dynamographdeployment glm47-nvfp4-1p1d
```

### View logs
```bash
# Frontend logs
kubectl logs -l component=frontend -f

# Worker logs
kubectl logs -l component=worker -l subcomponent=prefill -f
kubectl logs -l component=worker -l subcomponent=decode -f
```

### Test inference
```bash
# Port-forward the frontend service
kubectl port-forward service/glm47-nvfp4-1p1d-frontend 8000:8000

# Send test request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Performance Tuning

### For higher throughput
- Increase decode worker replicas (1P+2D or 1P+3D)
- Adjust `--mem-fraction-static` lower to allow more batch size
- Use `--enable-chunked-prefill` for very long prompts

### For lower latency
- Use single-node deployment (no disaggregation overhead)
- Reduce `--stream-interval` from 10 to 1
- Ensure workers on same physical rack (network locality)

### For stability
- Increase heartbeat timeout values (already set high)
- Monitor NCCL/NIXL transfer bandwidth
- Check GPU memory usage with `nvidia-smi`

## Troubleshooting

### Pods pending
```bash
# Check node resources
kubectl describe nodes | grep -A5 "Allocated resources"

# Check GPU availability
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPUS:.status.capacity.'nvidia\.com/gpu'
```

### NCCL errors
- Verify InfiniBand/network connectivity between nodes
- Check NCCL environment variables in pod spec
- Ensure shared memory size is sufficient (40Gi default)

### KV transfer failures
- NIXL backend requires proper network setup
- Check prefill → decode communication in logs
- Verify `--disaggregation-bootstrap-port 30001` is open

### OOM errors
- Reduce `--mem-fraction-static` to 0.75 or 0.7
- Check model size vs available GPU memory
- Monitor with `kubectl top pods`

## Scaling Guide

| Workload | Config | Nodes | GPUs | Expected Throughput |
|----------|--------|-------|------|---------------------|
| Development | Single | 1 | 4 | 300-500 tok/s |
| Production (moderate) | 1P+1D | 2 | 8 | ~2,100 tok/s |
| Production (high) | 1P+2D | 3 | 12 | ~4,000 tok/s |
| Production (very high) | 1P+3D | 4 | 16 | ~6,000 tok/s |

## References

- [NVIDIA Dynamo Documentation](https://docs.nvidia.com/ai-dynamo/)
- [SGLang Runtime Guide](https://github.com/sgl-project/sglang)
- [GLM-4 Model Card](https://huggingface.co/THUDM/glm-4-9b)
- [DeepSeek-R1 Example](https://github.com/ai-dynamo/dynamo/tree/main/recipes/deepseek-r1)
