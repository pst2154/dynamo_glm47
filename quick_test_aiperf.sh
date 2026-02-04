#!/bin/bash
# Quick AIPerf test for GLM-4.7-NVFP4
# Usage: ./quick_test_aiperf.sh [frontend_url]

FRONTEND_URL="${1:-http://nvl72d011-T01:8000}"
MODEL_NAME="glm-4.7"

echo "========================================"
echo "Quick AIPerf Test for GLM-4.7"
echo "Frontend: ${FRONTEND_URL}"
echo "Model: ${MODEL_NAME}"
echo "========================================"

# Quick test: 20 requests, concurrency 4
aiperf profile \
  --model ${MODEL_NAME} \
  --url ${FRONTEND_URL} \
  --endpoint-type chat \
  --streaming \
  --concurrency 4 \
  --request-count 20 \
  --output-tokens-mean 256 \
  --output-tokens-stddev 25 \
  --input-tokens-mean 1024 \
  --input-tokens-stddev 100 \
  --ui-type dashboard

echo ""
echo "Quick test complete!"
