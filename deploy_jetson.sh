#!/bin/bash
#*----------------------------------------------------------------------------*
#* Deployment script for Jetson Xavier NX                                    *
#* This script sets up the environment and deploys the EEGNet BCI system     *
#*----------------------------------------------------------------------------*

set -e  # Exit on error

echo "=========================================="
echo "EEGNet BCI Deployment for Jetson Xavier NX"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DEPLOY_DIR="$HOME/eegnet-bci"
MODEL_DIR="$DEPLOY_DIR/models"
DATA_DIR="$DEPLOY_DIR/data"

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo -e "${YELLOW}Warning: This script is designed for NVIDIA Jetson devices${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: Create directory structure
echo -e "\n${GREEN}[1/6] Creating directory structure...${NC}"
mkdir -p "$DEPLOY_DIR"
mkdir -p "$MODEL_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$DEPLOY_DIR/scripts"
echo "Deployment directory: $DEPLOY_DIR"

# Step 2: Copy necessary files
echo -e "\n${GREEN}[2/6] Copying files...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Copy Python modules
cp "$SCRIPT_DIR/models.py" "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/eeg_reduction.py" "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/inference.py" "$DEPLOY_DIR/"

# Copy deployment scripts
cp "$SCRIPT_DIR/requirements_jetson.txt" "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/DEPLOYMENT_GUIDE.md" "$DEPLOY_DIR/" 2>/dev/null || echo "Deployment guide not found, skipping"

# Make scripts executable
chmod +x "$DEPLOY_DIR/inference.py"

echo "Files copied successfully"

# Step 3: Check Python version
echo -e "\n${GREEN}[3/6] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION 3.6" | awk '{print ($1 >= $2)}') -eq 0 ]]; then
    echo -e "${YELLOW}Warning: Python 3.6+ recommended${NC}"
fi

# Step 4: Install system dependencies
echo -e "\n${GREEN}[4/6] Installing system dependencies...${NC}"
sudo apt-get update -qq
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-dev \
    zlib1g-dev \
    zip \
    libjpeg8-dev \
    liblapack-dev \
    libblas-dev \
    gfortran \
    > /dev/null 2>&1

echo "System dependencies installed"

# Step 5: Install Python packages
echo -e "\n${GREEN}[5/6] Installing Python packages...${NC}"

# Upgrade pip
pip3 install --upgrade pip setuptools wheel --quiet

# Check if TensorFlow is installed
if ! python3 -c "import tensorflow" 2>/dev/null; then
    echo -e "${YELLOW}TensorFlow not found. Installing TensorFlow for Jetson...${NC}"
    echo "Please install TensorFlow manually:"
    echo "  For JetPack 4.6:"
    echo "    sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow==1.15.5+nv20.12"
    echo "  For JetPack 5.x:"
    echo "    sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50 tensorflow"
    read -p "Press Enter after installing TensorFlow..."
fi

# Install other requirements
if [ -f "$DEPLOY_DIR/requirements_jetson.txt" ]; then
    pip3 install -r "$DEPLOY_DIR/requirements_jetson.txt" --quiet
    echo "Python packages installed"
else
    echo -e "${YELLOW}requirements_jetson.txt not found, skipping package installation${NC}"
fi

# Step 6: Verify installation
echo -e "\n${GREEN}[6/6] Verifying installation...${NC}"

# Check TensorFlow
if python3 -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)" 2>/dev/null; then
    echo -e "${GREEN}✓ TensorFlow installed${NC}"
else
    echo -e "${RED}✗ TensorFlow not found${NC}"
fi

# Check Keras
if python3 -c "import keras; print('Keras:', keras.__version__)" 2>/dev/null; then
    echo -e "${GREEN}✓ Keras installed${NC}"
else
    echo -e "${YELLOW}✗ Keras not found (may be included with TensorFlow)${NC}"
fi

# Check NumPy
if python3 -c "import numpy; print('NumPy:', numpy.__version__)" 2>/dev/null; then
    echo -e "${GREEN}✓ NumPy installed${NC}"
else
    echo -e "${RED}✗ NumPy not found${NC}"
fi

# Check if model files exist
if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR/*.h5 2>/dev/null)" ]; then
    MODEL_COUNT=$(ls -1 "$MODEL_DIR"/*.h5 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Found $MODEL_COUNT model file(s)${NC}"
else
    echo -e "${YELLOW}⚠ No model files found in $MODEL_DIR${NC}"
    echo "  Please copy your trained .h5 model files to: $MODEL_DIR"
fi

# Summary
echo -e "\n${GREEN}=========================================="
echo "Deployment completed!"
echo "==========================================${NC}"
echo ""
echo "Deployment directory: $DEPLOY_DIR"
echo "Model directory: $MODEL_DIR"
echo "Data directory: $DATA_DIR"
echo ""
echo "Next steps:"
echo "1. Copy your trained model files (.h5) to: $MODEL_DIR"
echo "2. Test inference with:"
echo "   cd $DEPLOY_DIR"
echo "   python3 inference.py --model models/your_model.h5 --data your_data.npy"
echo ""
echo "For performance optimization:"
echo "  sudo nvpmodel -m 0  # Maximum performance mode"
echo "  sudo jetson_clocks  # Set maximum clocks"
echo ""
