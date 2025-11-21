# EEGNet BCI Deployment Guide for Jetson Xavier NX

## Quick Start

### 1. Transfer Files to Jetson

**Option A: Using SCP (from your development machine)**
```bash
# Create deployment directory
ssh jetson@<jetson-ip> "mkdir -p ~/eegnet-bci"

# Transfer files
scp -r models.py eeg_reduction.py inference.py deploy_jetson.sh requirements_jetson.txt jetson@<jetson-ip>:~/eegnet-bci/
scp -r results/your-global-experiment/model/*.h5 jetson@<jetson-ip>:~/eegnet-bci/models/
```

**Option B: Using Git (if repository is available)**
```bash
ssh jetson@<jetson-ip>
cd ~
git clone <your-repo-url>
cd eegnet-based-embedded-bci
```

### 2. Run Deployment Script

```bash
cd ~/eegnet-bci  # or your deployment directory
chmod +x deploy_jetson.sh
./deploy_jetson.sh
```

### 3. Copy Model Files

Copy your trained model files to the models directory:
```bash
cp /path/to/your/models/*.h5 ~/eegnet-bci/models/
```

### 4. Test Inference

```bash
cd ~/eegnet-bci
python3 inference.py --model models/global_class_4_ds1_nch64_T3_split_0.h5 --data sample_data.npy
```

## Manual Installation

If you prefer manual installation:

### Step 1: Install System Dependencies

```bash
sudo apt-get update
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
    gfortran
```

### Step 2: Install TensorFlow for Jetson

**For JetPack 4.6 (TensorFlow 1.15):**
```bash
sudo pip3 install --pre --extra-index-url \
    https://developer.download.nvidia.com/compute/redist/jp/v46 \
    tensorflow==1.15.5+nv20.12
```

**For JetPack 5.x (TensorFlow 2.x):**
```bash
sudo pip3 install --extra-index-url \
    https://developer.download.nvidia.com/compute/redist/jp/v50 \
    tensorflow
```

### Step 3: Install Python Packages

```bash
pip3 install --upgrade pip setuptools wheel
pip3 install -r requirements_jetson.txt
```

### Step 4: Verify Installation

```bash
python3 -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python3 -c "import keras; print('Keras:', keras.__version__)"
python3 -c "import numpy; print('NumPy:', numpy.__version__)"
```

## Usage

### Basic Inference

```bash
python3 inference.py \
    --model models/global_class_4_ds1_nch64_T3_split_0.h5 \
    --data your_eeg_data.npy
```

### Ensemble Inference (Multiple Models)

```bash
python3 inference.py \
    --model "models/global_class_4_ds1_nch64_T3_split_*.h5" \
    --data your_eeg_data.npy \
    --ensemble
```

### Custom Parameters

```bash
python3 inference.py \
    --model models/your_model.h5 \
    --data your_data.npy \
    --n_ch 64 \
    --n_ds 1 \
    --T 3 \
    --batch_size 16 \
    --output predictions.npz
```

## Performance Optimization

### 1. Enable Maximum Performance Mode

```bash
# Set to maximum performance mode
sudo nvpmodel -m 0

# Set maximum clocks
sudo jetson_clocks

# Verify
sudo tegrastats
```

### 2. Monitor Performance

```bash
# Monitor GPU, CPU, memory usage
tegrastats

# Monitor in real-time
watch -n 1 tegrastats
```

### 3. Optimize Batch Size

For Jetson Xavier NX, recommended batch sizes:
- **Small models**: batch_size=16-32
- **Large models**: batch_size=8-16
- **Memory constrained**: batch_size=4-8

## Data Format

### Input Data Format

The inference script expects NumPy arrays with shape:
- **Raw data**: `(n_trials, n_channels, n_samples)`
- **After preprocessing**: `(n_trials, n_channels, n_samples, 1)`

Example:
```python
import numpy as np

# Create sample data: 10 trials, 64 channels, 480 samples (3 seconds at 160 Hz)
X = np.random.randn(10, 64, 480)
np.save('sample_data.npy', X)
```

### Model Parameters

- **n_ch**: Number of channels (8, 19, 27, 38, or 64)
- **n_ds**: Downsampling factor (1, 2, or 3)
- **T**: Time window in seconds (1, 2, or 3)
- **fs**: Sampling frequency (default: 160 Hz)

## Troubleshooting

### Issue: Out of Memory

**Solutions:**
- Reduce batch size: `--batch_size 4`
- Use smaller models or fewer channels
- Close unnecessary applications
- Check memory: `free -h`

### Issue: TensorFlow/Keras Version Mismatch

**Solutions:**
- Verify versions: `python3 -c "import tensorflow; import keras; print(tf.__version__, keras.__version__)"`
- Reinstall TensorFlow from NVIDIA repository
- Check JetPack version: `cat /etc/nv_tegra_release`

### Issue: Model Loading Fails

**Solutions:**
- Verify model file: `file models/*.h5`
- Check file permissions: `chmod 644 models/*.h5`
- Ensure all dependencies are installed
- Check model compatibility with TensorFlow version

### Issue: Slow Inference

**Solutions:**
- Enable performance mode: `sudo nvpmodel -m 0 && sudo jetson_clocks`
- Use TensorRT conversion (advanced)
- Reduce input data size if possible
- Check GPU utilization: `tegrastats`

### Issue: Import Errors

**Solutions:**
- Ensure all files are in the same directory
- Check Python path: `python3 -c "import sys; print(sys.path)"`
- Install missing packages: `pip3 install <package>`

## Real-Time Inference

For real-time EEG data acquisition, you'll need to:

1. **Connect EEG device** to Jetson (USB/Serial)
2. **Stream data** in real-time
3. **Buffer 3 seconds** of data (480 samples at 160 Hz)
4. **Preprocess** using `eeg_reduction()`
5. **Run inference** in a loop

Example real-time structure:
```python
import numpy as np
from inference import load_eegnet_model, preprocess_eeg_data, predict

# Load model
model = load_eegnet_model('models/your_model.h5')

# Real-time loop
while True:
    # Acquire 3 seconds of data (implement based on your hardware)
    eeg_data = acquire_eeg_data()  # Shape: (1, 64, 480)
    
    # Preprocess
    X = preprocess_eeg_data(eeg_data, n_ds=1, n_ch=64, T=3)
    
    # Predict
    predictions, classes = predict(model, X)
    
    # Use prediction
    print(f"Predicted class: {classes[0]}")
```

## File Structure

```
~/eegnet-bci/
├── models.py              # EEGNet model definition
├── eeg_reduction.py       # Data preprocessing
├── inference.py           # Inference script
├── requirements_jetson.txt # Python dependencies
├── deploy_jetson.sh      # Deployment script
├── models/                # Trained model files (.h5)
│   ├── global_class_4_ds1_nch64_T3_split_0.h5
│   └── ...
└── data/                  # Input/output data
    └── ...
```

## Additional Resources

- [NVIDIA Jetson Documentation](https://developer.nvidia.com/embedded/jetson-documentation)
- [TensorFlow on Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)
- [Jetson Performance Tuning](https://developer.nvidia.com/embedded/jetson-performance)
- [Jetson Xavier NX Developer Kit](https://developer.nvidia.com/embedded/jetson-xavier-nx-devkit)

## Support

For issues specific to this deployment:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Check Jetson system logs: `dmesg | tail -50`
4. Monitor system resources: `tegrastats`
