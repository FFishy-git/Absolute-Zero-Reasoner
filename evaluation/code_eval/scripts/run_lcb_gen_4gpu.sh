#!/bin/bash

# 4-GPU local launcher for LiveCodeBench code generation.
# Keeps the original 1-GPU wrapper untouched.

set -euo pipefail

MODEL_PATH="andrewzh/Absolute_Zero_Reasoner-Coder-7b"
CUDA_GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
N=1
TEMPERATURE=0.0
TOP_P=1.0
MAX_TOKENS=8096
BATCH_SIZE=128

while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--model)
      MODEL_PATH="$2"
      shift 2
      ;;
    -g|--gpu)
      CUDA_GPU_IDS="$2"
      shift 2
      ;;
    -n|--n)
      N="$2"
      shift 2
      ;;
    -t|--temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    -p|--top_p)
      TOP_P="$2"
      shift 2
      ;;
    -b|--batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    -k|--max_tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

NUM_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_GPU_IDS")

cd evaluation/code_eval/coding/LiveCodeBench

CUDA_VISIBLE_DEVICES="$CUDA_GPU_IDS" python -m lcb_runner.runner.main \
  --model "$MODEL_PATH" \
  --trust_remote_code \
  --scenario codegeneration \
  --release_version release_v5 \
  --tensor_parallel_size "$NUM_GPUS" \
  --use_cache \
  --n "$N" \
  --temperature "$TEMPERATURE" \
  --max_tokens "$MAX_TOKENS" \
  --custom_output_save_name "$MODEL_PATH" \
  --top_p "$TOP_P" \
  --timeout 60 \
  --evaluate --continue_existing --continue_existing_with_eval
