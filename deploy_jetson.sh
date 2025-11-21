#!/bin/bash
#*----------------------------------------------------------------------------*
#* Deployment script for Jetson Xavier NX                                     *
#* This script sets up the environment and installs dependencies              *
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

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo -e "${YELLOW}Warning: This script is designed for NVIDIA Jetson devices.${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Detect JetPack version
if [ -f /etc/nv_tegra_release ]; then
    JETPACK_VERSION=$(cat /etc/nv_tegra_release | head -c 3)
    echo -e "${GREEN}Detected JetPack version: ${JETPACK_VERSION}${NC}"
else
    JETPACK_VERSION="5.0"
    echo -e "${YELLOW}Could not detect JetPack version, assuming 5.0${NC}"
fi

# Update system
echo -e "\n${GREEN}[1/8] Updating system packages...${NC}"
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo -e "\n${GREEN}[2/8] Installing system dependencies...${NC}"
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
    git \
    wget

# Upgrade pip
echo -e "\n${GREEN}[3/8] Upgrading pip...${NC}"
python3 -m pip install --upgrade pip setuptools wheel

# Install TensorFlow for Jetson
echo -e "\n${GREEN}[4/8] Installing TensorFlow for Jetson...${NC}"
if [[ "$JETPACK_VERSION" == "4."* ]]; then
    echo "Installing TensorFlow 1.15 for JetPack 4.x"
    sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow==1.15.5+nv20.12
elif [[ "$JETPACK_VERSION" == "5."* ]]; then
    echo "Installing TensorFlow 2.x for JetPack 5.x"
    sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50 tensorflow
else
    echo -e "${YELLOW}Unknown JetPack version, attempting to install TensorFlow 2.x${NC}"
    sudo pip3 install tensorflow
fi

# Install Python dependencies
echo -e "\n${GREEN}[5/8] Installing Python dependencies...${NC}"
pip3 install numpy==1.18.5
pip3 install scipy==1.4.1
pip3 install scikit-learn==0.22.1
pip3 install h5py==2.10.0
pip3 install keras==2.2.4
pip3 install pyedflib==0.1.15

# Create project directory structure
echo -e "\n${GREEN}[6/8] Creating project directory structure...${NC}"
PROJECT_DIR="$HOME/eegnet-bci"
mkdir -p "$PROJECT_DIR"/{models,data,results,logs}

# Copy project files
echo -e "\n${GREEN}[7/8] Copying project files...${NC}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cp "$SCRIPT_DIR"/models.py "$PROJECT_DIR/"
cp "$SCRIPT_DIR"/eeg_reduction.py "$PROJECT_DIR/"
cp "$SCRIPT_DIR"/inference.py "$PROJECT_DIR/"
chmod +x "$PROJECT_DIR"/inference.py

# Set up performance mode
echo -e "\n${GREEN}[8/8] Configuring Jetson for maximum performance...${NC}"
echo -e "${YELLOW}Note: This requires sudo privileges${NC}"
read -p "Enable maximum performance mode? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo nvpmodel -m 0  # Maximum performance mode
    sudo jetson_clocks  # Set maximum clocks
    echo -e "${GREEN}Performance mode enabled!${NC}"
fi

# Verify installation
echo -e "\n${GREEN}Verifying installation...${NC}"
python3 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')" || echo -e "${RED}TensorFlow import failed!${NC}"
python3 -c "import keras; print(f'Keras version: {keras.__version__}')" || echo -e "${RED}Keras import failed!${NC}"
python3 -c "import numpy; print(f'NumPy version: {numpy.__version__}')" || echo -e "${RED}NumPy import failed!${NC}"

echo -e "\n${GREEN}=========================================="
echo "Deployment completed successfully!"
echo "==========================================${NC}"
echo -e "\nProject directory: ${GREEN}$PROJECT_DIR${NC}"
echo -e "\nNext steps:"
echo "1. Copy your trained model files (.h5) to: $PROJECT_DIR/models/"
echo "2. Test inference with: python3 $PROJECT_DIR/inference.py --model <model_path> --data <data_path>"
echo "3. For real-time inference: python3 $PROJECT_DIR/inference.py --model <model_path> --data <data_path> --realtime"
echo ""
echo -e "${YELLOW}Note: If you encounter BatchNormalization errors, see the README for fixes.${NC}"
