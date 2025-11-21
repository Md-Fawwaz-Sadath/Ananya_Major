# EEGNet Deployment Guide for Jetson Xavier NX

## Quick Start Guide

This guide will help you deploy your EEGNet model on NVIDIA Jetson Xavier NX using TensorRT for optimized inference.

## Prerequisites

### On Your Development Machine (where you have the .h5 model)
- Python 3.6+
- TensorFlow/Keras
- tf2onnx library

### On Jetson Xavier NX
- JetPack 4.6+ installed
- Python 3.6+
- TensorRT (pre-installed with JetPack)
- CUDA (pre-installed with JetPack)

---

## Step-by-Step Deployment Process

### Phase 1: Convert .h5 Model to ONNX (On Development Machine or Jetson)

#### 1.1 Install Required Dependencies
```bash
pip3 install tf2onnx onnx
```

#### 1.2 Convert Your .h5 Model to ONNX
```bash
# If you have a .h5 model file
python3 h5_to_onnx.py --input path/to/your/model.h5 --output model.onnx

# Example:
python3 h5_to_onnx.py --input results/your-global-experiment/model/global_class_4_ds1_nch64_T3_split_0.h5 --output output.onnx
```

**Note:** You already have `output.onnx` in your workspace! You can skip this step if you want to use the existing ONNX model.

---

### Phase 2: Transfer Files to Jetson Xavier NX

#### 2.1 Files to Transfer
Transfer the following files to your Jetson Xavier NX:

**Required:**
- `output.onnx` (or your converted ONNX model)
- `onnx_to_tensorrt.py`
- `run_eeg.py`
- `models.py`
- `eeg_reduction.py`

#### 2.2 Transfer Methods

**Method A: Using SCP (if connected via network)**
```bash
# From your development machine
scp output.onnx onnx_to_tensorrt.py run_eeg.py models.py eeg_reduction.py jetson@<jetson-ip>:~/eegnet/
```

**Method B: Using USB Drive**
1. Copy files to USB drive
2. Connect USB to Jetson
3. Copy files from USB to Jetson

**Method C: Using Git**
```bash
# On Jetson
git clone <your-repository-url>
cd <repository-name>
```

---

### Phase 3: Setup Jetson Xavier NX Environment

#### 3.1 Install System Dependencies
```bash
sudo apt-get update
sudo apt-get install -y python3-pip libhdf5-serial-dev hdf5-tools
```

#### 3.2 Install Python Dependencies
```bash
pip3 install numpy scipy
pip3 install pycuda  # Required for TensorRT
```

#### 3.3 Enable Maximum Performance Mode
```bash
# Set to maximum performance mode
sudo nvpmodel -m 0

# Set clocks to maximum
sudo jetson_clocks
```

---

### Phase 4: Convert ONNX to TensorRT (On Jetson)

#### 4.1 Convert to TensorRT Engine
```bash
cd ~/eegnet

# Convert ONNX to TensorRT with FP16 precision (faster)
python3 onnx_to_tensorrt.py --input output.onnx --output model.trt --fp16 --test

# Or without FP16 (higher precision)
python3 onnx_to_tensorrt.py --input output.onnx --output model.trt --no-fp16 --test
```

**Expected output:**
```
Building TensorRT engine (this may take a few minutes)...
✓ TensorRT engine built successfully
✓ Engine saved successfully
  Engine size: X.XX MB
```

This step will create a `model.trt` file optimized for your Jetson Xavier NX.

---

### Phase 5: Run Inference

#### 5.1 Prepare Your EEG Data
Your EEG data should be in one of these formats:
- `.npy` file: numpy array saved with `np.save()`
- `.npz` file: numpy archive saved with `np.savez()`
- `.csv` file: comma-separated values

**Expected data shape:** `(n_trials, 64, 480)` or `(64, 480)` for single trial
- 64 channels
- 480 time samples (3 seconds at 160 Hz)

#### 5.2 Run Inference with TensorRT
```bash
# Run inference with TensorRT engine (fastest)
python3 run_eeg.py \
    --model model.trt \
    --data your_eeg_data.npy \
    --num-classes 4

# With benchmarking
python3 run_eeg.py \
    --model model.trt \
    --data your_eeg_data.npy \
    --num-classes 4 \
    --benchmark \
    --num-iterations 100
```

#### 5.3 Alternative: Run with ONNX (without TensorRT)
```bash
# Install ONNX Runtime first
pip3 install onnxruntime-gpu

# Run inference
python3 run_eeg.py \
    --model output.onnx \
    --model-type onnx \
    --data your_eeg_data.npy \
    --num-classes 4
```

#### 5.4 Alternative: Run with Keras .h5 (slowest)
```bash
# If you have TensorFlow/Keras installed
pip3 install tensorflow keras h5py

python3 run_eeg.py \
    --model model.h5 \
    --model-type keras \
    --data your_eeg_data.npy \
    --num-classes 4
```

---

## Example Workflow

Here's a complete example from start to finish:

```bash
# On Jetson Xavier NX

# 1. Create working directory
mkdir -p ~/eegnet
cd ~/eegnet

# 2. Transfer files (assuming files are already here)
ls -lh
# Should see: output.onnx, onnx_to_tensorrt.py, run_eeg.py, etc.

# 3. Install dependencies
pip3 install numpy scipy pycuda

# 4. Enable max performance
sudo nvpmodel -m 0
sudo jetson_clocks

# 5. Convert to TensorRT
python3 onnx_to_tensorrt.py --input output.onnx --output model.trt --fp16

# 6. Create sample data for testing (if you don't have real data)
python3 << EOF
import numpy as np
# Create dummy EEG data: 5 trials, 64 channels, 480 samples
dummy_data = np.random.randn(5, 64, 480).astype(np.float32)
np.save('test_data.npy', dummy_data)
print("✓ Created test_data.npy")
EOF

# 7. Run inference
python3 run_eeg.py --model model.trt --data test_data.npy --num-classes 4

# 8. Run benchmark
python3 run_eeg.py --model model.trt --data test_data.npy --num-classes 4 --benchmark --num-iterations 100
```

---

## Expected Performance

### Inference Times (approximate)
- **TensorRT (FP16):** 1-5 ms per sample
- **TensorRT (FP32):** 2-8 ms per sample
- **ONNX Runtime:** 5-15 ms per sample
- **Keras/TensorFlow:** 10-30 ms per sample

### Throughput
- **TensorRT:** 200-1000 samples/second
- **ONNX:** 60-200 samples/second
- **Keras:** 30-100 samples/second

*Note: Actual performance depends on model size, input shape, and system load.*

---

## Preprocessing Parameters

The `run_eeg.py` script supports various preprocessing options:

```bash
python3 run_eeg.py \
    --model model.trt \
    --data your_data.npy \
    --channels 64 \           # Number of EEG channels (8, 19, 27, 38, or 64)
    --downsample 1 \          # Downsampling factor (1, 2, or 3)
    --time-window 3.0 \       # Time window in seconds (1, 2, or 3)
    --sampling-rate 160 \     # Sampling rate in Hz
    --num-classes 4           # Number of output classes (2, 3, or 4)
```

---

## Troubleshooting

### Issue: "ImportError: No module named 'tensorrt'"
**Solution:** TensorRT should be pre-installed with JetPack. Check:
```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"
```
If not found, ensure JetPack is properly installed.

### Issue: "pycuda._driver.Error: cuInit failed: no CUDA-capable device is detected"
**Solution:** 
```bash
# Check CUDA devices
nvidia-smi  # or jetson_release -v

# Ensure CUDA environment variables are set
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Issue: "Out of memory" during TensorRT conversion
**Solution:**
```bash
# Reduce workspace size
python3 onnx_to_tensorrt.py --input output.onnx --output model.trt --workspace 512

# Or close other applications to free memory
```

### Issue: Slow inference despite using TensorRT
**Solution:**
```bash
# Enable maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Check if GPU is being used
tegrastats  # Monitor GPU usage
```

### Issue: "ONNX model check failed"
**Solution:**
```bash
# Verify ONNX model
python3 << EOF
import onnx
model = onnx.load('output.onnx')
onnx.checker.check_model(model)
print("Model is valid!")
EOF
```

---

## Model Variants

If you trained multiple model variants, you can convert and test them:

```bash
# Convert multiple models
for split in {0..4}; do
    python3 h5_to_onnx.py \
        --input results/your-global-experiment/model/global_class_4_ds1_nch64_T3_split_${split}.h5 \
        --output model_split_${split}.onnx
    
    python3 onnx_to_tensorrt.py \
        --input model_split_${split}.onnx \
        --output model_split_${split}.trt \
        --fp16
done

# Test each model
for split in {0..4}; do
    echo "Testing split ${split}..."
    python3 run_eeg.py \
        --model model_split_${split}.trt \
        --data test_data.npy \
        --num-classes 4
done
```

---

## Real-Time Streaming

For real-time EEG data streaming from a device:

1. Collect 3 seconds of data (480 samples at 160 Hz)
2. Format as numpy array: shape `(1, 64, 480)`
3. Pass to `run_eeg.py`

Example pseudo-code:
```python
import numpy as np
from run_eeg import EEGInferenceEngine, preprocess_data

# Initialize engine once
engine = EEGInferenceEngine('model.trt', 'tensorrt')

# Streaming loop
while True:
    # Get 3 seconds of data from your EEG device
    raw_data = get_eeg_data_from_device()  # shape: (64, 480)
    
    # Add batch dimension
    data = np.expand_dims(raw_data, axis=0)
    
    # Preprocess
    data_preprocessed = preprocess_data(data, n_ds=1, n_ch=64, T=3)
    
    # Predict
    prediction = engine.predict(data_preprocessed)
    predicted_class = np.argmax(prediction)
    
    print(f"Predicted class: {predicted_class}")
```

---

## Directory Structure

Recommended directory structure on Jetson:

```
~/eegnet/
├── output.onnx              # ONNX model
├── model.trt                # TensorRT engine
├── onnx_to_tensorrt.py      # Conversion script
├── run_eeg.py               # Inference script
├── models.py                # Model architecture
├── eeg_reduction.py         # Preprocessing functions
├── test_data.npy            # Test data
└── predictions.npy          # Output predictions
```

---

## Next Steps

1. ✅ Convert your .h5 model to ONNX (or use existing output.onnx)
2. ✅ Transfer files to Jetson Xavier NX
3. ✅ Convert ONNX to TensorRT on Jetson
4. ✅ Run inference with your EEG data
5. ✅ Optimize performance with FP16 precision
6. ✅ Integrate with your real-time EEG acquisition system

---

## Additional Resources

- [NVIDIA Jetson Documentation](https://developer.nvidia.com/embedded/jetson-documentation)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [ONNX Documentation](https://onnx.ai/onnx/)

---

## Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure you're using the correct model format and data shape
4. Check Jetson system resources with `tegrastats`

---

**Good luck with your deployment! 🚀**
