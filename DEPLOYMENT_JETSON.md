# EEGNet BCI Deployment Guide for Jetson Xavier NX

This guide provides step-by-step instructions for deploying the EEGNet-based Brain-Computer Interface (BCI) system on NVIDIA Jetson Xavier NX.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Detailed Setup](#detailed-setup)
4. [Running Inference](#running-inference)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)
7. [Real-Time Deployment](#real-time-deployment)

## Prerequisites

### Hardware Requirements
- **NVIDIA Jetson Xavier NX** (or compatible Jetson device)
- **MicroSD card** (64GB+ recommended, Class 10 or better)
- **Power supply** (official Jetson power adapter, 5V/4A)
- **Cooling solution** (fan or heatsink recommended)
- **EEG acquisition device** (compatible with Jetson via USB/Serial)

### Software Requirements
- **JetPack SDK 4.6+** (for TensorFlow 1.x) or **JetPack 5.x** (for TensorFlow 2.x)
- **Python 3.6+**
- **Internet connection** (for downloading dependencies)

## Quick Start

### Option 1: Automated Deployment (Recommended)

1. **Transfer files to Jetson:**
   ```bash
   # On your development machine
   scp -r /workspace/* jetson@<jetson-ip>:~/eegnet-deploy/
   ```

2. **SSH into Jetson:**
   ```bash
   ssh jetson@<jetson-ip>
   ```

3. **Run deployment script:**
   ```bash
   cd ~/eegnet-deploy
   chmod +x deploy_jetson.sh
   ./deploy_jetson.sh
   ```

4. **Copy model files:**
   ```bash
   # Copy your trained .h5 model files to the models directory
   cp /path/to/your/models/*.h5 ~/eegnet-bci/models/
   ```

5. **Test inference:**
   ```bash
   cd ~/eegnet-bci
   python3 inference.py --model models/global_class_4_ds1_nch64_T3_split_0.h5 --data your_data.npy
   ```

### Option 2: Manual Deployment

Follow the [Detailed Setup](#detailed-setup) section below.

## Detailed Setup

### Step 1: Flash JetPack

1. Download **JetPack SDK** from [NVIDIA Developer website](https://developer.nvidia.com/embedded/jetpack)
   - For TensorFlow 1.15: Use **JetPack 4.6.x**
   - For TensorFlow 2.x: Use **JetPack 5.x**
2. Flash the Jetson using NVIDIA SDK Manager or balenaEtcher
3. Complete initial setup (username, password, WiFi, etc.)

### Step 2: Update System

```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-dev
```

### Step 3: Install System Dependencies

```bash
sudo apt-get install -y \
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
```

### Step 4: Install Python Package Manager

```bash
python3 -m pip install --upgrade pip setuptools wheel
```

### Step 5: Install TensorFlow for Jetson

**For JetPack 4.6 (TensorFlow 1.15):**
```bash
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow==1.15.5+nv20.12
```

**For JetPack 5.x (TensorFlow 2.x):**
```bash
sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50 tensorflow
```

### Step 6: Install Python Dependencies

```bash
pip3 install -r requirements_jetson.txt
```

Or install individually:
```bash
pip3 install numpy==1.18.5
pip3 install scipy==1.4.1
pip3 install scikit-learn==0.22.1
pip3 install h5py==2.10.0
pip3 install keras==2.2.4
pip3 install pyedflib==0.1.15  # Only if reading EDF files
```

### Step 7: Verify Installation

```bash
python3 -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python3 -c "import keras; print('Keras:', keras.__version__)"
python3 -c "import numpy; print('NumPy:', numpy.__version__)"
```

### Step 8: Set Up Project Directory

```bash
mkdir -p ~/eegnet-bci/{models,data,results,logs}
cd ~/eegnet-bci
```

### Step 9: Copy Project Files

Copy the following files to `~/eegnet-bci/`:
- `models.py` - EEGNet model definition
- `eeg_reduction.py` - Data preprocessing
- `inference.py` - Inference script
- `output.onnx` - ONNX model (optional, for ONNX Runtime)

```bash
cp /path/to/deployment/models.py ~/eegnet-bci/
cp /path/to/deployment/eeg_reduction.py ~/eegnet-bci/
cp /path/to/deployment/inference.py ~/eegnet-bci/
chmod +x ~/eegnet-bci/inference.py
```

### Step 10: Copy Model Files

Copy your trained model files (`.h5` format) to the models directory:

```bash
cp /path/to/trained/models/*.h5 ~/eegnet-bci/models/
```

## Running Inference

### Basic Inference

```bash
cd ~/eegnet-bci
python3 inference.py \
    --model models/global_class_4_ds1_nch64_T3_split_0.h5 \
    --data your_eeg_data.npy
```

### Real-Time Inference (Single Trial)

```bash
python3 inference.py \
    --model models/global_class_4_ds1_nch64_T3_split_0.h5 \
    --data your_eeg_data.npy \
    --realtime
```

### Batch Inference with Output

```bash
python3 inference.py \
    --model models/global_class_4_ds1_nch64_T3_split_0.h5 \
    --data your_eeg_data.npy \
    --output results/predictions.npz \
    --batch-size 16
```

### Benchmark Mode

```bash
python3 inference.py \
    --model models/global_class_4_ds1_nch64_T3_split_0.h5 \
    --data your_eeg_data.npy \
    --benchmark
```

### Command-Line Options

```
--model          Path to trained model (.h5 file) [required]
--data           Path to input data (.npy or .npz file) [required]
--output         Path to save predictions (optional)
--batch-size     Batch size for inference (default: 16)
--num-classes    Number of classes (default: 4)
--n-ds           Downsampling factor (default: 1)
--n-ch           Number of channels (default: 64)
--T              Time window in seconds (default: 3.0)
--realtime       Run in real-time mode (single trial inference)
--benchmark      Run benchmark test
```

## Performance Optimization

### 1. Enable Maximum Performance Mode

```bash
sudo nvpmodel -m 0  # Maximum performance mode
sudo jetson_clocks  # Set maximum clocks
```

To make this permanent:
```bash
sudo systemctl enable nvpmodel
```

### 2. Monitor Performance

```bash
# Monitor system stats
tegrastats

# Monitor GPU usage
sudo tegrastats --interval 1000
```

### 3. Optimize Batch Size

For Jetson Xavier NX, recommended batch sizes:
- **Small models**: batch_size = 16-32
- **Large models**: batch_size = 8-16
- **Real-time**: batch_size = 1

### 4. Memory Management

Jetson Xavier NX has 8GB unified memory. Monitor usage:
```bash
free -h
tegrastats
```

If running out of memory:
- Reduce batch size
- Close unnecessary applications
- Use model quantization

### 5. TensorRT Optimization (Advanced)

For faster inference, convert models to TensorRT:

```bash
# Install TensorRT (usually pre-installed with JetPack)
# Convert .h5 to TensorRT engine
# (Requires additional conversion script)
```

## Troubleshooting

### Issue: BatchNormalization Error

**Error:** `TypeError: 'NoneType' object is not callable` in BatchNormalization

**Solution:**
1. Edit TensorFlow backend file:
   ```bash
   sudo nano /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py
   ```
2. Change lines 1908, 1910, 1914, 1918:
   - Change `beta = tf.reshape(beta, (-1))` to `beta = tf.reshape(beta, [-1])`
   - Change `gamma = tf.reshape(gamma, (-1))` to `gamma = tf.reshape(gamma, [-1])`
3. Ensure `channels_last` in Keras config:
   ```bash
   python3 -c "from keras import backend as K; print(K.image_data_format())"
   # Should output: channels_last
   ```

### Issue: Out of Memory

**Solution:**
- Reduce batch size: `--batch-size 8`
- Limit GPU memory in `inference.py` (already configured)
- Close other applications
- Use model quantization

### Issue: TensorFlow/Keras Version Mismatch

**Solution:**
```bash
# Uninstall and reinstall
pip3 uninstall tensorflow keras
# Then reinstall according to JetPack version (see Step 5)
```

### Issue: Slow Inference

**Solution:**
1. Enable performance mode: `sudo nvpmodel -m 0 && sudo jetson_clocks`
2. Check GPU utilization: `tegrastats`
3. Verify TensorFlow GPU support:
   ```python
   import tensorflow as tf
   print(tf.test.is_gpu_available())
   ```
4. Use appropriate batch size
5. Consider TensorRT conversion

### Issue: Model Loading Fails

**Solution:**
- Verify model file integrity: `file models/*.h5`
- Check file permissions: `chmod 644 models/*.h5`
- Ensure all dependencies are installed
- Check model compatibility with TensorFlow version

### Issue: Import Errors

**Solution:**
```bash
# Verify Python path
python3 -c "import sys; print(sys.path)"

# Reinstall problematic packages
pip3 install --force-reinstall <package-name>
```

## Real-Time Deployment

For real-time EEG inference, you need to:

1. **Connect EEG device** to Jetson (USB/Serial)
2. **Stream data** in real-time
3. **Buffer 3 seconds** of data (480 samples at 160 Hz)
4. **Preprocess** using `eeg_reduction()`
5. **Run inference** continuously

### Example Real-Time Loop

```python
import numpy as np
from inference import EEGNetInference

# Initialize inference engine
engine = EEGNetInference('models/your_model.h5')

# Real-time loop
while True:
    # Acquire 3 seconds of EEG data (480 samples at 160 Hz)
    eeg_data = acquire_eeg_data(duration=3.0)  # Shape: (64, 480)
    
    # Run inference
    result = engine.predict_single(eeg_data)
    
    # Process result
    print(f"Class: {result['class']}, Confidence: {result['confidence']:.2f}")
    
    # Control output or feedback
    # ...
```

### Data Format Requirements

- **Input shape**: `(n_channels, n_samples)` for single trial
- **Channels**: 64 (or 8, 19, 27, 38 based on configuration)
- **Sampling rate**: 160 Hz
- **Time window**: 3 seconds (480 samples)
- **Data type**: float32

## File Structure

After deployment, your directory should look like:

```
~/eegnet-bci/
├── models.py              # EEGNet model definition
├── eeg_reduction.py       # Data preprocessing
├── inference.py           # Inference script
├── models/                # Trained model files
│   ├── global_class_4_ds1_nch64_T3_split_0.h5
│   ├── global_class_4_ds1_nch64_T3_split_1.h5
│   └── ...
├── data/                  # Input data files
├── results/               # Output predictions
└── logs/                  # Log files
```

## Additional Resources

- [NVIDIA Jetson Documentation](https://developer.nvidia.com/embedded/jetson-documentation)
- [TensorFlow on Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)
- [Jetson Performance Tuning](https://developer.nvidia.com/embedded/jetson-performance)
- [Jetson Xavier NX Datasheet](https://developer.nvidia.com/embedded/jetson-xavier-nx)

## Support

For issues specific to this deployment:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review error messages carefully
3. Verify all dependencies are correctly installed
4. Check Jetson system logs: `dmesg | tail -50`

## License

Please refer to the LICENSE file for licensing information.
