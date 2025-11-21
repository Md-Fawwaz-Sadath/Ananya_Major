#!/bin/bash

# Create minimal deployment package for Jetson Xavier NX
# This creates a tarball with only essential files

echo "========================================================================"
echo "Creating Minimal Deployment Package for Jetson Xavier NX"
echo "========================================================================"
echo ""

# Create temporary directory
DEPLOY_DIR="eegnet_jetson_minimal"
mkdir -p "$DEPLOY_DIR"

echo "Copying essential files..."

# Copy only essential Python scripts
cp output.onnx "$DEPLOY_DIR/" 2>/dev/null || echo "Warning: output.onnx not found"
cp onnx_to_tensorrt.py "$DEPLOY_DIR/"
cp run_eeg.py "$DEPLOY_DIR/"
cp models.py "$DEPLOY_DIR/"
cp eeg_reduction.py "$DEPLOY_DIR/"
cp create_test_data.py "$DEPLOY_DIR/"

# Copy cleanup and deployment scripts
cp cleanup_jetson.sh "$DEPLOY_DIR/"
cp jetson_deploy.sh "$DEPLOY_DIR/"

# Copy documentation
cp JETSON_QUICKSTART.md "$DEPLOY_DIR/"
cp JETSON_CLEANUP_GUIDE.md "$DEPLOY_DIR/" 2>/dev/null || true

# Create README
cat > "$DEPLOY_DIR/README.txt" << 'EOF'
EEGNET JETSON XAVIER NX MINIMAL DEPLOYMENT PACKAGE
===================================================

This package contains only the essential files needed for deploying
EEGNet on Jetson Xavier NX.

CONTENTS:
---------
1. output.onnx              - Pre-converted ONNX model
2. onnx_to_tensorrt.py      - Convert ONNX to TensorRT
3. run_eeg.py               - Inference script
4. models.py                - Model architecture
5. eeg_reduction.py         - Data preprocessing
6. create_test_data.py      - Generate test data
7. cleanup_jetson.sh        - Clean Jetson storage
8. jetson_deploy.sh         - Automated deployment
9. JETSON_QUICKSTART.md     - Quick start guide

TOTAL SIZE: < 50 KB (plus model files)

QUICK START:
------------
1. Transfer this package to Jetson:
   scp -r eegnet_jetson_minimal jetson@<jetson-ip>:~/

2. On Jetson, run cleanup (if needed):
   cd ~/eegnet_jetson_minimal
   chmod +x cleanup_jetson.sh
   ./cleanup_jetson.sh

3. Deploy:
   chmod +x jetson_deploy.sh
   ./jetson_deploy.sh

4. Or manually:
   pip3 install numpy scipy pycuda
   python3 onnx_to_tensorrt.py --input output.onnx --output model.trt --fp16
   python3 create_test_data.py
   python3 run_eeg.py --model model.trt --data test_data.npy --num-classes 4

STORAGE REQUIREMENTS:
---------------------
- Scripts: < 50 KB
- Python packages (numpy, scipy, pycuda): ~200 MB
- TensorRT engine: ~0.1-1 MB
- Test data: ~5-10 MB
- Total: < 250 MB

For detailed instructions, see JETSON_QUICKSTART.md

EOF

# Make scripts executable
chmod +x "$DEPLOY_DIR"/*.sh
chmod +x "$DEPLOY_DIR"/*.py

echo ""
echo "Creating tarball..."
tar -czf eegnet_jetson_minimal.tar.gz "$DEPLOY_DIR"

# Get package size
PACKAGE_SIZE=$(du -h eegnet_jetson_minimal.tar.gz | cut -f1)

echo ""
echo "========================================================================"
echo "Deployment Package Created Successfully!"
echo "========================================================================"
echo ""
echo "Package: eegnet_jetson_minimal.tar.gz"
echo "Size: $PACKAGE_SIZE"
echo ""
echo "Contents:"
ls -lh "$DEPLOY_DIR"
echo ""
echo "To transfer to Jetson:"
echo "  scp eegnet_jetson_minimal.tar.gz jetson@<jetson-ip>:~/"
echo ""
echo "On Jetson, extract with:"
echo "  tar -xzf eegnet_jetson_minimal.tar.gz"
echo "  cd eegnet_jetson_minimal"
echo "  ./jetson_deploy.sh"
echo ""
echo "========================================================================"
echo ""

# Cleanup
read -p "Remove temporary directory? (y/n): " cleanup
if [ "$cleanup" = "y" ] || [ "$cleanup" = "Y" ]; then
    rm -rf "$DEPLOY_DIR"
    echo "Temporary directory removed"
fi

echo ""
echo "✅ Done!"
