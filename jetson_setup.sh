#!/bin/bash
################################################################################
# Jetson Xavier NX Setup Script for EEGNet BCI Deployment
# 
# This script automates the installation of dependencies for running
# EEGNet-based motor imagery classification on Jetson Xavier NX
#
# Copyright (C) 2020 ETH Zurich, Switzerland
# SPDX-License-Identifier: Apache-2.0
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================================"
echo "  EEGNet BCI - Jetson Xavier NX Setup Script"
echo "================================================================"
echo ""

# Function to print colored messages
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if running on Jetson
check_jetson() {
    if [ ! -f /etc/nv_tegra_release ]; then
        print_warning "Not running on Jetson device. Continue anyway? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_status "Detected Jetson device"
        cat /etc/nv_tegra_release
    fi
}

# Update system
update_system() {
    print_status "Updating system packages..."
    sudo apt-get update
    sudo apt-get upgrade -y
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
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
        cmake \
        libopenblas-dev \
        libopenmpi-dev
    
    print_status "System dependencies installed"
}

# Upgrade pip
upgrade_pip() {
    print_status "Upgrading pip..."
    python3 -m pip install --upgrade pip setuptools wheel
}

# Detect JetPack version
detect_jetpack() {
    print_status "Detecting JetPack version..."
    
    if command -v jetson_release &> /dev/null; then
        jetson_release
    else
        print_warning "jetson_release not found. Installing..."
        sudo pip3 install jetson-stats
        print_status "Run 'jetson_release' to see your JetPack version"
    fi
}

# Install TensorFlow for Jetson
install_tensorflow() {
    print_status "Installing TensorFlow for Jetson..."
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_status "Python version: $PYTHON_VERSION"
    
    # Install dependencies for TensorFlow
    sudo apt-get install -y \
        libhdf5-serial-dev \
        hdf5-tools \
        libhdf5-dev \
        zlib1g-dev \
        zip \
        libjpeg8-dev \
        liblapack-dev \
        libblas-dev \
        gfortran
    
    # Install NumPy and other dependencies first
    pip3 install --user numpy==1.19.4
    pip3 install --user future==0.18.2
    pip3 install --user mock==3.0.5
    pip3 install --user keras_preprocessing==1.1.2
    pip3 install --user keras_applications==1.0.8
    pip3 install --user gast==0.4.0
    pip3 install --user protobuf==3.19.6
    pip3 install --user pybind11
    pip3 install --user cython
    pip3 install --user h5py==3.1.0
    
    # Install TensorFlow for JetPack 4.6
    print_status "Installing TensorFlow 2.7.0 for JetPack 4.6..."
    pip3 install --user --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow==2.7.0+nv22.1
    
    print_status "TensorFlow installation complete"
}

# Install Keras
install_keras() {
    print_status "Installing Keras..."
    pip3 install --user keras==2.6.0
    print_status "Keras installed"
}

# Install ONNX Runtime with CUDA support
install_onnxruntime() {
    print_status "Installing ONNX Runtime with CUDA support..."
    
    # Install dependencies
    pip3 install --user cmake
    
    # Install ONNX Runtime GPU for Jetson
    # Note: Pre-built wheels may not be available, so we provide CPU version
    pip3 install --user onnxruntime-gpu==1.12.1 || {
        print_warning "ONNX Runtime GPU not available, installing CPU version..."
        pip3 install --user onnxruntime==1.12.1
    }
    
    print_status "ONNX Runtime installed"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    pip3 install --user scipy==1.7.3
    pip3 install --user scikit-learn==1.0.2
    pip3 install --user matplotlib==3.5.1
    pip3 install --user Pillow==9.0.1
    
    print_status "Python dependencies installed"
}

# Enable max performance mode
enable_max_performance() {
    print_status "Setting Jetson to maximum performance mode..."
    
    # Set to max performance mode (mode 0)
    sudo nvpmodel -m 0
    
    # Enable max clocks
    sudo jetson_clocks
    
    print_status "Maximum performance mode enabled"
    print_warning "Note: This will increase power consumption and heat generation"
}

# Create project directory structure
setup_project_structure() {
    print_status "Setting up project directory..."
    
    mkdir -p ~/eegnet-bci
    mkdir -p ~/eegnet-bci/models
    mkdir -p ~/eegnet-bci/data
    mkdir -p ~/eegnet-bci/logs
    
    print_status "Project directory created at ~/eegnet-bci"
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    echo ""
    echo "Testing Python packages..."
    
    python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')" || print_error "NumPy failed"
    python3 -c "import scipy; print(f'SciPy: {scipy.__version__}')" || print_error "SciPy failed"
    python3 -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')" || print_error "scikit-learn failed"
    python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" || print_error "TensorFlow failed"
    python3 -c "import keras; print(f'Keras: {keras.__version__}')" || print_error "Keras failed"
    python3 -c "import onnxruntime as ort; print(f'ONNX Runtime: {ort.__version__}')" || print_error "ONNX Runtime failed"
    
    echo ""
    print_status "Checking CUDA availability..."
    python3 -c "import tensorflow as tf; print('CUDA available:', tf.test.is_built_with_cuda())"
    python3 -c "import tensorflow as tf; print('GPU devices:', tf.config.list_physical_devices('GPU'))"
    
    echo ""
    print_status "Verification complete!"
}

# Display next steps
display_next_steps() {
    echo ""
    echo "================================================================"
    echo "  Installation Complete!"
    echo "================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Copy your model files to ~/eegnet-bci/models/"
    echo "  2. Copy eeg_reduction.py and models.py to ~/eegnet-bci/"
    echo "  3. Copy jetson_inference.py to ~/eegnet-bci/"
    echo ""
    echo "Run inference:"
    echo "  cd ~/eegnet-bci"
    echo "  python3 jetson_inference.py --model models/output.onnx --mode demo"
    echo ""
    echo "Run benchmark:"
    echo "  python3 jetson_inference.py --model models/output.onnx --mode benchmark"
    echo ""
    echo "Monitor system performance:"
    echo "  tegrastats"
    echo "  jtop  # If you have jetson-stats installed"
    echo ""
    echo "================================================================"
}

# Main installation flow
main() {
    echo "Starting installation..."
    echo ""
    
    # Check if we should skip certain steps
    SKIP_UPDATE=false
    SKIP_TENSORFLOW=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-update)
                SKIP_UPDATE=true
                shift
                ;;
            --skip-tensorflow)
                SKIP_TENSORFLOW=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --skip-update       Skip system update"
                echo "  --skip-tensorflow   Skip TensorFlow installation"
                echo "  --help              Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    check_jetson
    
    if [ "$SKIP_UPDATE" = false ]; then
        update_system
    fi
    
    install_system_deps
    upgrade_pip
    detect_jetpack
    
    if [ "$SKIP_TENSORFLOW" = false ]; then
        install_tensorflow
        install_keras
    fi
    
    install_onnxruntime
    install_python_deps
    setup_project_structure
    
    # Ask about performance mode
    echo ""
    print_warning "Enable maximum performance mode? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        enable_max_performance
    fi
    
    verify_installation
    display_next_steps
}

# Run main function
main "$@"
