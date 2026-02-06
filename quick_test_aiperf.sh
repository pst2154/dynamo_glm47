#!/bin/bash
# Quick AIPerf test for GLM-4.7-NVFP4
# Usage: ./quick_test_aiperf.sh [frontend_url]

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_ENV="${SCRIPT_DIR}/../python_env"
if [[ -f "${PYTHON_ENV}/bin/activate" ]]; then
  source "${PYTHON_ENV}/bin/activate"
fi

FRONTEND_URL="${1:-http://nvl72d012-T17:8000}"
MODEL_NAME="glm-4.7"
# Local tokenizer so AIPerf doesn't try to fetch "glm-4.7" from Hugging Face
TOKENIZER_PATH="${SCRIPT_DIR}/../GLM-4.7-NVFP4"
# Disable reasoning/thinking for benchmarks
EXTRA_INPUTS='{"chat_template_args":{"enable_thinking":false}}'

echo "========================================"
echo "Quick AIPerf Test for GLM-4.7"
echo "Frontend: ${FRONTEND_URL}"
echo "Model: ${MODEL_NAME}"
echo "========================================"

# Quick test: 20 requests, concurrency 4
aiperf profile \
  --model ${MODEL_NAME} \
  --tokenizer "${TOKENIZER_PATH}" \
  --url ${FRONTEND_URL} \
  --endpoint-type chat \
  --streaming \
  --concurrency 4 \
  --request-count 20 \
  --output-tokens-mean 256 \
  --output-tokens-stddev 25 \
  --synthetic-input-tokens-mean 1024 \
  --synthetic-input-tokens-stddev 100 \
  --extra-inputs "${EXTRA_INPUTS}" \
  --ui-type dashboard

echo ""
echo "Quick test complete!"
