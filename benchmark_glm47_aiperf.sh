#!/bin/bash
# AIPerf Benchmark Script for GLM-4.7-NVFP4 (4P+2D Disaggregated Setup)
# Usage: ./benchmark_glm47_aiperf.sh [frontend_url]

set -e

# Configuration
FRONTEND_URL="${1:-http://localhost:8000}"
MODEL_NAME="glm-4.7"
OUTPUT_DIR="./aiperf_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "AIPerf Benchmark for GLM-4.7-NVFP4"
echo "========================================"
echo "Frontend URL: ${FRONTEND_URL}"
echo "Model: ${MODEL_NAME}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo "========================================"
echo ""

# Test 1: Quick warmup and connectivity test
echo "=== Test 1: Warmup (10 requests, concurrency 1) ==="
aiperf profile \
  --model ${MODEL_NAME} \
  --url ${FRONTEND_URL} \
  --endpoint-type chat \
  --streaming \
  --concurrency 1 \
  --request-count 10 \
  --output-tokens-mean 100 \
  --output-tokens-stddev 10 \
  --input-tokens-mean 512 \
  --input-tokens-stddev 50 \
  --artifact-dir "${OUTPUT_DIR}/01_warmup_${TIMESTAMP}" \
  --ui-type simple

echo ""
echo "=== Test 2: Low Concurrency (50 requests, concurrency 4) ==="
aiperf profile \
  --model ${MODEL_NAME} \
  --url ${FRONTEND_URL} \
  --endpoint-type chat \
  --streaming \
  --concurrency 4 \
  --request-count 50 \
  --output-tokens-mean 256 \
  --output-tokens-stddev 25 \
  --input-tokens-mean 1024 \
  --input-tokens-stddev 100 \
  --artifact-dir "${OUTPUT_DIR}/02_concurrency_4_${TIMESTAMP}" \
  --ui-type dashboard

echo ""
echo "=== Test 3: Medium Concurrency (100 requests, concurrency 8) ==="
aiperf profile \
  --model ${MODEL_NAME} \
  --url ${FRONTEND_URL} \
  --endpoint-type chat \
  --streaming \
  --concurrency 8 \
  --request-count 100 \
  --output-tokens-mean 256 \
  --output-tokens-stddev 25 \
  --input-tokens-mean 1024 \
  --input-tokens-stddev 100 \
  --artifact-dir "${OUTPUT_DIR}/03_concurrency_8_${TIMESTAMP}" \
  --ui-type dashboard

echo ""
echo "=== Test 4: High Concurrency (100 requests, concurrency 16) ==="
aiperf profile \
  --model ${MODEL_NAME} \
  --url ${FRONTEND_URL} \
  --endpoint-type chat \
  --streaming \
  --concurrency 16 \
  --request-count 100 \
  --output-tokens-mean 256 \
  --output-tokens-stddev 25 \
  --input-tokens-mean 1024 \
  --input-tokens-stddev 100 \
  --artifact-dir "${OUTPUT_DIR}/04_concurrency_16_${TIMESTAMP}" \
  --ui-type dashboard

echo ""
echo "=== Test 5: Request Rate Test (60 seconds, 10 req/sec) ==="
aiperf profile \
  --model ${MODEL_NAME} \
  --url ${FRONTEND_URL} \
  --endpoint-type chat \
  --streaming \
  --request-rate 10 \
  --benchmark-duration 60 \
  --output-tokens-mean 256 \
  --output-tokens-stddev 25 \
  --input-tokens-mean 1024 \
  --input-tokens-stddev 100 \
  --artifact-dir "${OUTPUT_DIR}/05_request_rate_10_${TIMESTAMP}" \
  --ui-type dashboard

echo ""
echo "=== Test 6: Long Context Test (20 requests, concurrency 4, 4K tokens) ==="
aiperf profile \
  --model ${MODEL_NAME} \
  --url ${FRONTEND_URL} \
  --endpoint-type chat \
  --streaming \
  --concurrency 4 \
  --request-count 20 \
  --output-tokens-mean 512 \
  --output-tokens-stddev 50 \
  --input-tokens-mean 4096 \
  --input-tokens-stddev 100 \
  --artifact-dir "${OUTPUT_DIR}/06_long_context_4k_${TIMESTAMP}" \
  --ui-type dashboard

echo ""
echo "========================================"
echo "Benchmark Complete!"
echo "========================================"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "To view results:"
echo "  cd ${OUTPUT_DIR}"
echo "  ls -lrt"
echo ""
echo "To generate comparison plots:"
echo "  aiperf plot --input-dirs ${OUTPUT_DIR}/*/"
echo "========================================"
