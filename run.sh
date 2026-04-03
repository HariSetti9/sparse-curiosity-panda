#!/bin/bash

# Sparse Rewards with Intrinsic Curiosity - One-Command Execution Script
# This script installs dependencies, trains models, evaluates, and visualizes results.

set -e  # Exit on error

echo "============================================================"
echo "Sparse Rewards with Intrinsic Curiosity for Robotic Reaching"
echo "============================================================"
echo ""

# Step 1: Install dependencies
echo "[1/5] Installing dependencies..."
pip install -q -r requirements.txt

# Step 2: Create necessary directories
echo "[2/5] Creating directories..."
mkdir -p models logs results

# Step 3: Train both baseline and sparse+curiosity methods
echo "[3/5] Training models (this may take ~40 minutes)..."
python train.py

# Step 4: Evaluate trained models
echo "[4/5] Evaluating models..."
python evaluate.py

# Step 5: Generate visualizations
echo "[5/5] Generating visualizations..."
python visualize.py

echo ""
echo "============================================================"
echo "Pipeline Complete!"
echo "============================================================"
echo ""
echo "Generated files:"
echo "  Models:"
ls -lh models/*.zip 2>/dev/null || echo "    (no models found)"
echo ""
echo "  Logs:"
ls -lh logs/*.json 2>/dev/null || echo "    (no logs found)"
echo ""
echo "  Results:"
ls -lh results/ 2>/dev/null || echo "    (no results found)"
echo ""
echo "To view the evaluation report:"
echo "  cat results/evaluation_report.txt"
echo ""
echo "To regenerate visualizations:"
echo "  python visualize.py"
echo ""
