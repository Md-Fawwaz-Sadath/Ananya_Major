#!/bin/bash
################################################################################
# Create Deployment Package for Jetson Xavier NX
# 
# This script packages all necessary files for deployment on Jetson Xavier NX
#
# Copyright (C) 2020 ETH Zurich, Switzerland
# SPDX-License-Identifier: Apache-2.0
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================"
echo "  EEGNet BCI - Creating Deployment Package for Jetson Xavier NX"
echo "================================================================"
echo ""

# Package name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PACKAGE_NAME="eegnet_jetson_deployment_${TIMESTAMP}"
PACKAGE_DIR="./${PACKAGE_NAME}"

# Create package directory
echo -e "${GREEN}Creating package directory...${NC}"
mkdir -p "${PACKAGE_DIR}"
mkdir -p "${PACKAGE_DIR}/models"
mkdir -p "${PACKAGE_DIR}/docs"

# Copy essential files
echo -e "${GREEN}Copying essential files...${NC}"

# Core Python files
if [ -f "eeg_reduction.py" ]; then
    cp eeg_reduction.py "${PACKAGE_DIR}/"
    echo "  ✓ eeg_reduction.py"
else
    echo -e "${YELLOW}  ⚠ eeg_reduction.py not found${NC}"
fi

if [ -f "models.py" ]; then
    cp models.py "${PACKAGE_DIR}/"
    echo "  ✓ models.py"
else
    echo -e "${YELLOW}  ⚠ models.py not found${NC}"
fi

# Inference script
if [ -f "jetson_inference.py" ]; then
    cp jetson_inference.py "${PACKAGE_DIR}/"
    echo "  ✓ jetson_inference.py"
else
    echo -e "${YELLOW}  ⚠ jetson_inference.py not found${NC}"
fi

# Setup script
if [ -f "jetson_setup.sh" ]; then
    cp jetson_setup.sh "${PACKAGE_DIR}/"
    chmod +x "${PACKAGE_DIR}/jetson_setup.sh"
    echo "  ✓ jetson_setup.sh"
else
    echo -e "${YELLOW}  ⚠ jetson_setup.sh not found${NC}"
fi

# Requirements
if [ -f "requirements_jetson.txt" ]; then
    cp requirements_jetson.txt "${PACKAGE_DIR}/"
    echo "  ✓ requirements_jetson.txt"
else
    echo -e "${YELLOW}  ⚠ requirements_jetson.txt not found${NC}"
fi

# Model files
echo ""
echo -e "${GREEN}Copying model files...${NC}"

if [ -f "output.onnx" ]; then
    cp output.onnx "${PACKAGE_DIR}/models/"
    echo "  ✓ output.onnx"
else
    echo -e "${YELLOW}  ⚠ output.onnx not found${NC}"
fi

# Check for .h5 models
H5_COUNT=$(find . -maxdepth 2 -name "*.h5" 2>/dev/null | wc -l)
if [ "$H5_COUNT" -gt 0 ]; then
    echo "  Found $H5_COUNT .h5 model files"
    find . -maxdepth 2 -name "*.h5" -exec cp {} "${PACKAGE_DIR}/models/" \;
    echo "  ✓ Copied .h5 models"
fi

# Documentation
echo ""
echo -e "${GREEN}Copying documentation...${NC}"

if [ -f "DEPLOYMENT_JETSON.md" ]; then
    cp DEPLOYMENT_JETSON.md "${PACKAGE_DIR}/docs/"
    echo "  ✓ DEPLOYMENT_JETSON.md"
fi

if [ -f "README.md" ]; then
    cp README.md "${PACKAGE_DIR}/docs/"
    echo "  ✓ README.md"
fi

if [ -f "LICENSE" ]; then
    cp LICENSE "${PACKAGE_DIR}/"
    echo "  ✓ LICENSE"
fi

# Create quick start guide
echo ""
echo -e "${GREEN}Creating QUICKSTART.txt...${NC}"
cat > "${PACKAGE_DIR}/QUICKSTART.txt" << 'EOF'
EEGNet BCI - Jetson Xavier NX Deployment
=========================================

QUICK START GUIDE
-----------------

1. TRANSFER FILES TO JETSON
   
   Option A - SCP (from development machine):
   $ scp -r eegnet_jetson_deployment_* jetson@<jetson-ip>:~/
   
   Option B - USB Drive:
   - Copy this folder to USB drive
   - Mount on Jetson and copy to home directory

2. RUN SETUP SCRIPT
   
   $ cd ~/eegnet_jetson_deployment_*/
   $ chmod +x jetson_setup.sh
   $ ./jetson_setup.sh
   
   This will install all dependencies (~30-60 minutes)

3. RUN INFERENCE DEMO
   
   $ cd ~/eegnet_jetson_deployment_*/
   $ python3 jetson_inference.py --model models/output.onnx --mode demo

4. BENCHMARK PERFORMANCE
   
   $ python3 jetson_inference.py --model models/output.onnx --mode benchmark

5. ENABLE MAX PERFORMANCE (Optional)
   
   $ sudo nvpmodel -m 0
   $ sudo jetson_clocks

TROUBLESHOOTING
---------------

If you encounter issues, see docs/DEPLOYMENT_JETSON.md for detailed instructions.

Common commands:
- Check GPU: tegrastats
- Monitor system: jtop (install: sudo pip3 install jetson-stats)
- Check TensorFlow: python3 -c "import tensorflow as tf; print(tf.__version__)"

DIRECTORY STRUCTURE
-------------------

eegnet_jetson_deployment_*/
├── models/
│   └── output.onnx              # ONNX model
├── docs/
│   ├── DEPLOYMENT_JETSON.md     # Full deployment guide
│   └── README.md                # Original README
├── eeg_reduction.py             # Preprocessing
├── models.py                    # Model architecture
├── jetson_inference.py          # Inference script
├── jetson_setup.sh              # Setup script
├── requirements_jetson.txt      # Python dependencies
└── QUICKSTART.txt               # This file

For detailed documentation, see docs/DEPLOYMENT_JETSON.md

Copyright (C) 2020 ETH Zurich, Switzerland
SPDX-License-Identifier: Apache-2.0
EOF

echo "  ✓ QUICKSTART.txt created"

# Create transfer script
echo ""
echo -e "${GREEN}Creating transfer_to_jetson.sh...${NC}"
cat > "${PACKAGE_DIR}/transfer_to_jetson.sh" << 'EOF'
#!/bin/bash
# Transfer files to Jetson Xavier NX via SCP

if [ -z "$1" ]; then
    echo "Usage: ./transfer_to_jetson.sh <jetson-ip-address>"
    echo "Example: ./transfer_to_jetson.sh 192.168.1.100"
    exit 1
fi

JETSON_IP=$1
JETSON_USER="jetson"  # Change if different

echo "Transferring files to ${JETSON_USER}@${JETSON_IP}..."

# Create directory on Jetson
ssh ${JETSON_USER}@${JETSON_IP} "mkdir -p ~/eegnet-bci"

# Transfer files
scp -r ./* ${JETSON_USER}@${JETSON_IP}:~/eegnet-bci/

echo ""
echo "Transfer complete!"
echo ""
echo "Next steps:"
echo "1. SSH to Jetson: ssh ${JETSON_USER}@${JETSON_IP}"
echo "2. Run setup: cd ~/eegnet-bci && ./jetson_setup.sh"
EOF

chmod +x "${PACKAGE_DIR}/transfer_to_jetson.sh"
echo "  ✓ transfer_to_jetson.sh created"

# Get package size
PACKAGE_SIZE=$(du -sh "${PACKAGE_DIR}" | cut -f1)

# Create archive
echo ""
echo -e "${GREEN}Creating compressed archive...${NC}"
tar -czf "${PACKAGE_NAME}.tar.gz" "${PACKAGE_DIR}"
ARCHIVE_SIZE=$(du -sh "${PACKAGE_NAME}.tar.gz" | cut -f1)

echo ""
echo "================================================================"
echo "  Deployment Package Created Successfully!"
echo "================================================================"
echo ""
echo "Package directory: ${PACKAGE_DIR}"
echo "Package size: ${PACKAGE_SIZE}"
echo "Archive: ${PACKAGE_NAME}.tar.gz"
echo "Archive size: ${ARCHIVE_SIZE}"
echo ""
echo "Contents:"
ls -lh "${PACKAGE_DIR}"
echo ""
echo "Models:"
ls -lh "${PACKAGE_DIR}/models/" 2>/dev/null || echo "  (no models found)"
echo ""
echo "DEPLOYMENT OPTIONS:"
echo ""
echo "Option 1 - Transfer directory:"
echo "  scp -r ${PACKAGE_DIR} jetson@<jetson-ip>:~/"
echo ""
echo "Option 2 - Transfer archive:"
echo "  scp ${PACKAGE_NAME}.tar.gz jetson@<jetson-ip>:~/"
echo "  ssh jetson@<jetson-ip> 'tar -xzf ${PACKAGE_NAME}.tar.gz'"
echo ""
echo "Option 3 - Use transfer script:"
echo "  cd ${PACKAGE_DIR}"
echo "  ./transfer_to_jetson.sh <jetson-ip>"
echo ""
echo "Option 4 - USB Drive:"
echo "  Copy ${PACKAGE_DIR} or ${PACKAGE_NAME}.tar.gz to USB drive"
echo ""
echo "================================================================"
echo ""
echo "Once transferred to Jetson, run:"
echo "  cd ~/eegnet-bci  # or ~/${PACKAGE_NAME}"
echo "  ./jetson_setup.sh"
echo ""
echo "For detailed instructions, see docs/DEPLOYMENT_JETSON.md"
echo "================================================================"
