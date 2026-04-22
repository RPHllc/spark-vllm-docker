#!/bin/bash
# Install the GB10 tuned MoE kernel config into vLLM's configs directory.
#
# Generated 2026-04-21 via vLLM's benchmarks/kernels/benchmark_moe.py on GB10
# hardware for Qwen/Qwen3.6-35B-A3B-FP8 (E=256, N_half=512, hidden=2048,
# topk=8, dtype=fp8_w8a8, block_shape=[128,128]). Covers batch sizes
# [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256].
#
# Without this, vLLM falls back to a generic Triton MoE kernel and prints
# "Using default MoE config. Performance might be sub-optimal!", which was
# more than 20% slower on this hardware during local testing.
set -e

MOD_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIGS_DIR=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs
CONFIG_NAME='E=256,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json'
SOURCE_CONFIG="$MOD_DIR/$CONFIG_NAME"
TARGET_CONFIG="$CONFIGS_DIR/$CONFIG_NAME"

if [[ ! -f "$SOURCE_CONFIG" ]]; then
  echo "Missing tuned MoE config: $SOURCE_CONFIG" >&2
  exit 1
fi

mkdir -p "$CONFIGS_DIR"
install -m 0644 "$SOURCE_CONFIG" "$TARGET_CONFIG"
echo "=====> Installed tuned MoE kernel config for Qwen3.6-35B-A3B-FP8 on GB10"
