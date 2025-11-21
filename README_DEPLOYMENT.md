# Quick Deployment Guide for Jetson Xavier NX

## 🚀 Quick Start (5 minutes)

### Step 1: Transfer Files to Jetson

From your development machine:
```bash
# Replace <jetson-ip> with your Jetson's IP address
JETSON_IP="<jetson-ip>"  # e.g., "192.168.1.100"
JETSON_USER="jetson"     # or your username

# Create directory on Jetson
ssh $JETSON_USER@$JETSON_IP "mkdir -p ~/eegnet-bci/models"

# Transfer deployment files
scp inference.py eeg_reduction.py models.py deploy_jetson.sh requirements_jetson.txt \
    $JETSON_USER@$JETSON_IP:~/eegnet-bci/

# Transfer model files (if you have them)
scp results/*/model/*.h5 $JETSON_USER@$JETSON_IP:~/eegnet-bci/models/ 2>/dev/null || echo "No models found, skip this step"
```

### Step 2: Run Deployment Script on Jetson

SSH into your Jetson:
```bash
ssh $JETSON_USER@$JETSON_IP
cd ~/eegnet-bci
chmod +x deploy_jetson.sh
./deploy_jetson.sh
```

### Step 3: Test Deployment

```bash
cd ~/eegnet-bci

# Create test data
python3 test_deployment.py --create-data --data-file test_data.npy

# Test inference (if you have a model)
python3 inference.py --model models/your_model.h5 --data test_data.npy
```

## 📋 What Was Created

### Core Files
- **`inference.py`** - Main inference script (TensorFlow/Keras)
- **`inference_onnx.py`** - ONNX Runtime inference (faster, optional)
- **`test_deployment.py`** - Test script to verify installation
- **`deploy_jetson.sh`** - Automated deployment script
- **`requirements_jetson.txt`** - Python dependencies
- **`DEPLOYMENT_GUIDE.md`** - Detailed deployment guide

### Original Files (Required)
- **`models.py`** - EEGNet model definition
- **`eeg_reduction.py`** - Data preprocessing functions

## 🔧 Manual Setup (Alternative)

If the automated script doesn't work:

```bash
# 1. Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev libhdf5-serial-dev hdf5-tools

# 2. Install TensorFlow for Jetson
# For JetPack 4.6:
sudo pip3 install --pre --extra-index-url \
    https://developer.download.nvidia.com/compute/redist/jp/v46 \
    tensorflow==1.15.5+nv20.12

# 3. Install Python packages
pip3 install -r requirements_jetson.txt
```

## 📊 Usage Examples

### Basic Inference
```bash
python3 inference.py --model models/model.h5 --data data.npy
```

### Ensemble Prediction (Multiple Models)
```bash
python3 inference.py \
    --model "models/global_class_4_ds1_nch64_T3_split_*.h5" \
    --data data.npy \
    --ensemble
```

### ONNX Inference (Faster)
```bash
# First install ONNX Runtime
pip3 install onnxruntime-gpu

# Then use ONNX inference
python3 inference_onnx.py --model output.onnx --data data.npy
```

## ⚡ Performance Optimization

```bash
# Enable maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Monitor performance
tegrastats
```

## 📁 Directory Structure

```
~/eegnet-bci/
├── inference.py           # Main inference script
├── inference_onnx.py     # ONNX inference (optional)
├── test_deployment.py    # Test script
├── models.py             # Model definition
├── eeg_reduction.py      # Preprocessing
├── deploy_jetson.sh      # Deployment script
├── requirements_jetson.txt
├── models/               # Put your .h5 model files here
│   └── *.h5
└── data/                 # Input/output data
    └── *.npy
```

## 🐛 Troubleshooting

### "TensorFlow not found"
```bash
# Install TensorFlow for your JetPack version
# Check JetPack version:
cat /etc/nv_tegra_release

# Install accordingly (see DEPLOYMENT_GUIDE.md)
```

### "Out of memory"
```bash
# Reduce batch size
python3 inference.py --model model.h5 --data data.npy --batch_size 4

# Or enable performance mode
sudo nvpmodel -m 0 && sudo jetson_clocks
```

### "Model file not found"
```bash
# Copy your trained models to:
cp /path/to/models/*.h5 ~/eegnet-bci/models/
```

## 📚 More Information

- **Detailed Guide**: See `DEPLOYMENT_GUIDE.md`
- **Original README**: See `README.md`
- **Jetson Docs**: https://developer.nvidia.com/embedded/jetson-documentation

## ✅ Verification Checklist

- [ ] Files transferred to Jetson
- [ ] Deployment script run successfully
- [ ] TensorFlow installed and working
- [ ] Model files copied to `models/` directory
- [ ] Test script runs: `python3 test_deployment.py`
- [ ] Inference works: `python3 inference.py --model <model.h5> --data <data.npy>`
- [ ] Performance mode enabled: `sudo nvpmodel -m 0 && sudo jetson_clocks`

## 🎯 Next Steps

1. **Copy your trained models** to `~/eegnet-bci/models/`
2. **Prepare your EEG data** in NumPy format (.npy)
3. **Run inference** on real data
4. **Integrate with your EEG hardware** for real-time inference
5. **Optimize performance** using TensorRT (advanced)

---

**Need Help?** Check `DEPLOYMENT_GUIDE.md` for detailed troubleshooting and advanced options.
