#!/bin/bash

CHECKPOINT_DIR="./checkpoints"
MODEL_PATH="$CHECKPOINT_DIR/depth_pro.pt"
MODEL_URL="https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"

mkdir -p "$CHECKPOINT_DIR"

if [ -f "$MODEL_PATH" ]; then
    echo "✅ Model already exists at $MODEL_PATH — skipping download."
else
    echo "⬇️  Downloading DepthPro model..."
    wget "$MODEL_URL" -P "$MODEL_PATH"
    echo "✅ DepthPro model downloaded successfully."
fi
