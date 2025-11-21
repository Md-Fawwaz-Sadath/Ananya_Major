# Jetson Xavier NX Quick Start Guide

## 🚀 Quick Deployment in 3 Steps

### Step 1: Transfer Files to Jetson
Transfer these files to your Jetson Xavier NX:
```
output.onnx
onnx_to_tensorrt.py
run_eeg.py
models.py
eeg_reduction.py
jetson_deploy.sh
```

**Using SCP:**
```bash
scp output.onnx onnx_to_tensorrt.py run_eeg.py models.py eeg_reduction.py jetson_deploy.sh jetson@<jetson-ip>:~/eegnet/
```

### Step 2: Run Automated Deployment Script
On your Jetson Xavier NX:
```bash
cd ~/eegnet
chmod +x jetson_deploy.sh
./jetson_deploy.sh
```

This script will:
- ✓ Check dependencies
- ✓ Install required packages
- ✓ Enable maximum performance mode
- ✓ Convert ONNX to TensorRT
- ✓ Create test data
- ✓ Run inference test

### Step 3: Run Inference with Your Data
```bash
python3 run_eeg.py --model model.trt --data your_eeg_data.npy --num-classes 4
```

---

## 📋 Manual Deployment (Alternative)

If you prefer manual control:

### 1. Install Dependencies
```bash
pip3 install numpy scipy pycuda
```

### 2. Enable Max Performance
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### 3. Convert to TensorRT
```bash
python3 onnx_to_tensorrt.py --input output.onnx --output model.trt --fp16
```

### 4. Run Inference
```bash
python3 run_eeg.py --model model.trt --data your_data.npy --num-classes 4
```

---

## 📊 Expected Performance

| Model Type | Inference Time | Throughput |
|------------|---------------|------------|
| TensorRT (FP16) | 1-5 ms | 200-1000 samples/sec |
| TensorRT (FP32) | 2-8 ms | 125-500 samples/sec |
| ONNX Runtime | 5-15 ms | 60-200 samples/sec |

---

## 🔧 Usage Examples

### Basic Inference
```bash
python3 run_eeg.py --model model.trt --data test_data.npy --num-classes 4
```

### With Benchmark
```bash
python3 run_eeg.py \
    --model model.trt \
    --data test_data.npy \
    --num-classes 4 \
    --benchmark \
    --num-iterations 100
```

### Custom Preprocessing
```bash
python3 run_eeg.py \
    --model model.trt \
    --data your_data.npy \
    --channels 64 \
    --downsample 1 \
    --time-window 3.0 \
    --sampling-rate 160 \
    --num-classes 4
```

### Save Predictions
```bash
python3 run_eeg.py \
    --model model.trt \
    --data your_data.npy \
    --num-classes 4 \
    --output predictions.npy
```

---

## 📁 Required Data Format

Your EEG data should be:
- **Format:** `.npy`, `.npz`, or `.csv`
- **Shape:** `(n_trials, 64, 480)` or `(64, 480)` for single trial
  - 64 channels (or 8, 19, 27, 38)
  - 480 time samples (3 seconds at 160 Hz)
- **Type:** `float32` or `float64`

**Create test data:**
```bash
python3 << EOF
import numpy as np
data = np.random.randn(5, 64, 480).astype(np.float32)
np.save('my_data.npy', data)
EOF
```

---

## ❓ Troubleshooting

### "No module named 'tensorrt'"
TensorRT is pre-installed with JetPack. If missing:
```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

### "Out of memory"
Reduce workspace size:
```bash
python3 onnx_to_tensorrt.py --input output.onnx --output model.trt --workspace 512
```

### Slow inference
Enable max performance:
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Check GPU usage
```bash
tegrastats
```

---

## 📚 File Descriptions

| File | Purpose |
|------|---------|
| `output.onnx` | ONNX model (already converted) |
| `onnx_to_tensorrt.py` | Convert ONNX to TensorRT |
| `run_eeg.py` | Run inference |
| `models.py` | Model architecture |
| `eeg_reduction.py` | Data preprocessing |
| `jetson_deploy.sh` | Automated deployment script |
| `test_onnx_model.py` | Test ONNX model validity |
| `h5_to_onnx.py` | Convert .h5 to ONNX (if needed) |

---

## 🔄 If You Have .h5 Model Files

If you need to convert .h5 files to ONNX:

```bash
# Install tf2onnx
pip3 install tf2onnx onnx

# Convert
python3 h5_to_onnx.py --input model.h5 --output model.onnx

# Then follow normal deployment
python3 onnx_to_tensorrt.py --input model.onnx --output model.trt --fp16
```

---

## ✅ Quick Checklist

- [ ] Files transferred to Jetson
- [ ] Dependencies installed (`pip3 install numpy scipy pycuda`)
- [ ] Max performance enabled (`sudo nvpmodel -m 0 && sudo jetson_clocks`)
- [ ] ONNX converted to TensorRT (`python3 onnx_to_tensorrt.py ...`)
- [ ] Test data prepared (`.npy` file with correct shape)
- [ ] Inference tested (`python3 run_eeg.py ...`)
- [ ] Performance benchmarked (with `--benchmark` flag)

---

## 🎯 Complete Example Session

```bash
# On Jetson Xavier NX
mkdir -p ~/eegnet && cd ~/eegnet

# (Transfer files here)

# Install deps
pip3 install numpy scipy pycuda

# Max performance
sudo nvpmodel -m 0 && sudo jetson_clocks

# Convert to TensorRT
python3 onnx_to_tensorrt.py --input output.onnx --output model.trt --fp16

# Create test data
python3 -c "import numpy as np; np.save('test.npy', np.random.randn(5,64,480).astype('f'))"

# Run inference
python3 run_eeg.py --model model.trt --data test.npy --num-classes 4

# Benchmark
python3 run_eeg.py --model model.trt --data test.npy --num-classes 4 --benchmark
```

---

## 📞 Need Help?

See full documentation: `DEPLOYMENT_GUIDE.md`

---

**Happy Deploying! 🎉**
