#!/usr/bin/env python3
# Patch SGLang for GLM-4.7-NVFP4 compatibility
# See: https://huggingface.co/Tengyunw/GLM-4.7-NVFP4

import sglang
import os

quant_file = os.path.join(os.path.dirname(sglang.__file__), 'srt/layers/quantization/modelopt_quant.py')
print(f"Patching: {quant_file}")

with open(quant_file, 'r') as f:
    content = f.read()

# Replace the divisibility check to always pass
content = content.replace(
    'weight_scale.shape[assert_dim] % 16 == 0',
    'True  # GLM-4.7-NVFP4 patch'
)

with open(quant_file, 'w') as f:
    f.write(content)

print("Patch applied!")
