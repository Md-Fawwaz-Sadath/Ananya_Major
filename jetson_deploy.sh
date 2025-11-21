#!/bin/bash

# Jetson Xavier NX Deployment Script
# This script automates the deployment process on Jetson Xavier NX

set -e  # Exit on error

echo "========================================================================"
echo "EEGNet Jetson Xavier NX Deployment Script"
echo "========================================================================"
echo ""

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "⚠ Warning: This doesn't appear to be a Jetson device"
    echo "This script is designed for NVIDIA Jetson Xavier NX"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python3
echo "Checking dependencies..."
if ! command_exists python3; then
    echo "✗ Python3 not found. Please install it first."
    exit 1
fi
echo "✓ Python3 found: $(python3 --version)"

# Check ONNX model
if [ ! -f "output.onnx" ]; then
    echo "✗ output.onnx not found in current directory"
    echo "Please ensure the ONNX model is in the current directory"
    exit 1
fi
echo "✓ ONNX model found: output.onnx"
echo ""

# Install Python dependencies
echo "========================================================================"
echo "Installing Python Dependencies"
echo "========================================================================"
echo ""

echo "Installing numpy and scipy..."
pip3 install --user numpy scipy 2>&1 | grep -v "Requirement already satisfied" || true

echo "Installing pycuda (required for TensorRT)..."
pip3 install --user pycuda 2>&1 | grep -v "Requirement already satisfied" || true

echo ""
echo "✓ Dependencies installed"
echo ""

# Enable maximum performance mode
echo "========================================================================"
echo "Configuring Jetson Performance"
echo "========================================================================"
echo ""

echo "Enabling maximum performance mode..."
if command_exists nvpmodel; then
    sudo nvpmodel -m 0
    echo "✓ Power mode set to maximum"
else
    echo "⚠ nvpmodel not found, skipping power mode configuration"
fi

if command_exists jetson_clocks; then
    sudo jetson_clocks
    echo "✓ Clocks set to maximum"
else
    echo "⚠ jetson_clocks not found, skipping clock configuration"
fi

echo ""

# Test ONNX model
echo "========================================================================"
echo "Testing ONNX Model"
echo "========================================================================"
echo ""

if [ -f "test_onnx_model.py" ]; then
    python3 test_onnx_model.py output.onnx
else
    echo "⚠ test_onnx_model.py not found, skipping model test"
fi

echo ""

# Convert to TensorRT
echo "========================================================================"
echo "Converting to TensorRT"
echo "========================================================================"
echo ""

if [ -f "onnx_to_tensorrt.py" ]; then
    read -p "Convert ONNX to TensorRT? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Converting to TensorRT (this may take a few minutes)..."
        python3 onnx_to_tensorrt.py --input output.onnx --output model.trt --fp16 --test
        echo ""
        echo "✓ TensorRT engine created: model.trt"
    else
        echo "Skipping TensorRT conversion"
    fi
else
    echo "✗ onnx_to_tensorrt.py not found"
    exit 1
fi

echo ""

# Create test data
echo "========================================================================"
echo "Creating Test Data"
echo "========================================================================"
echo ""

if [ ! -f "test_data.npy" ]; then
    echo "Creating dummy test data..."
    python3 << EOF
import numpy as np
# Create dummy EEG data: 5 trials, 64 channels, 480 samples
dummy_data = np.random.randn(5, 64, 480).astype(np.float32)
np.save('test_data.npy', dummy_data)
print("✓ Created test_data.npy")
EOF
else
    echo "✓ test_data.npy already exists"
fi

echo ""

# Run inference test
echo "========================================================================"
echo "Running Inference Test"
echo "========================================================================"
echo ""

if [ -f "model.trt" ] && [ -f "run_eeg.py" ]; then
    read -p "Run inference test? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Running inference..."
        python3 run_eeg.py --model model.trt --data test_data.npy --num-classes 4
        echo ""
        echo "✓ Inference test completed"
    fi
else
    echo "⚠ Required files not found for inference test"
fi

echo ""

# Summary
echo "========================================================================"
echo "Deployment Complete!"
echo "========================================================================"
echo ""
echo "Files created:"
echo "  - model.trt (TensorRT engine)"
echo "  - test_data.npy (test data)"
echo ""
echo "To run inference with your own data:"
echo "  python3 run_eeg.py --model model.trt --data your_data.npy --num-classes 4"
echo ""
echo "To run benchmark:"
echo "  python3 run_eeg.py --model model.trt --data test_data.npy --num-classes 4 --benchmark"
echo ""
echo "For more information, see DEPLOYMENT_GUIDE.md"
echo "========================================================================"
