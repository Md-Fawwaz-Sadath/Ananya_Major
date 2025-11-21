# Jetson Xavier NX Deployment Guide (Ubuntu)
## For users already ON the Jetson

You are here: `nvidia@ubuntu:~$` (Jetson Xavier NX)

---

## STEP 1: Get the Deployment Files

### Option A: Get Files from Development Machine

**First, on development machine, create a simple transfer package:**

```bash
# On development machine (/workspace):
cd /workspace
tar -czf jetson_deploy_simple.tar.gz \
  output.onnx \
  eeg_reduction.py \
  models.py \
  jetson_inference.py \
  jetson_setup.sh \
  requirements_jetson.txt

# Check your Jetson's IP (run on Jetson):
# hostname -I

# Transfer (run on development machine):
scp jetson_deploy_simple.tar.gz nvidia@<jetson-ip>:~/
```

**Then on Jetson (where you are now):**

```bash
cd ~
tar -xzf jetson_deploy_simple.tar.gz
ls -la
```

### Option B: Manual Setup (If can't transfer)

If you can't transfer files, I'll create them directly on your Jetson. Let me know!

---

## STEP 2: Install System Dependencies (Ubuntu on Jetson)

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install build tools and libraries
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    pkg-config \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-dev \
    zlib1g-dev \
    zip \
    libjpeg8-dev \
    liblapack-dev \
    libblas-dev \
    libopenblas-dev \
    gfortran \
    libatlas-base-dev \
    libfreetype6-dev \
    libpng-dev

# Upgrade pip
python3 -m pip install --upgrade pip setuptools wheel
```

---

## STEP 3: Check JetPack Version

```bash
# Check JetPack version
cat /etc/nv_tegra_release

# Check CUDA
ls /usr/local/cuda*/

# Check Python version
python3 --version

# Check available disk space (need ~3GB)
df -h
```

**Expected output:**
```
# Release "32.x.x", REVISION: x.x - JetPack 4.6.x
# OR
# Release "35.x.x", REVISION: x.x - JetPack 5.x
```

---

## STEP 4: Install TensorFlow for Jetson

### For JetPack 4.6 (L4T R32.x):

```bash
# Install dependencies
pip3 install --user numpy==1.19.4
pip3 install --user future==0.18.2
pip3 install --user mock==3.0.5
pip3 install --user keras_preprocessing==1.1.2
pip3 install --user keras_applications==1.0.8
pip3 install --user gast==0.4.0
pip3 install --user protobuf==3.19.6
pip3 install --user h5py==3.1.0
pip3 install --user cython

# Install TensorFlow
pip3 install --user --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow==2.7.0+nv22.1

# Install Keras
pip3 install --user keras==2.6.0
```

### For JetPack 5.x (L4T R35.x):

```bash
# Install dependencies
pip3 install --user numpy
pip3 install --user h5py

# Install TensorFlow 2.x
pip3 install --user --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50 tensorflow==2.11.0+nv23.3

# Install Keras
pip3 install --user keras
```

---

## STEP 5: Install ONNX Runtime

```bash
# Install ONNX Runtime
pip3 install --user onnxruntime==1.12.1

# Or if GPU version available:
pip3 install --user onnxruntime-gpu==1.12.1 || pip3 install --user onnxruntime==1.12.1
```

---

## STEP 6: Install Additional Dependencies

```bash
# Install scientific computing libraries
pip3 install --user scipy==1.7.3
pip3 install --user scikit-learn==1.0.2

# Optional: for visualization
pip3 install --user matplotlib
```

---

## STEP 7: Verify Installation

```bash
# Test imports
python3 << 'EOF'
import sys
print("Python:", sys.version)

try:
    import numpy
    print("✓ NumPy:", numpy.__version__)
except:
    print("✗ NumPy failed")

try:
    import tensorflow as tf
    print("✓ TensorFlow:", tf.__version__)
    print("  CUDA available:", tf.test.is_built_with_cuda())
    print("  GPU devices:", tf.config.list_physical_devices('GPU'))
except Exception as e:
    print("✗ TensorFlow failed:", e)

try:
    import keras
    print("✓ Keras:", keras.__version__)
except:
    print("✗ Keras failed")

try:
    import onnxruntime
    print("✓ ONNX Runtime:", onnxruntime.__version__)
    print("  Providers:", onnxruntime.get_available_providers())
except:
    print("✗ ONNX Runtime failed")

try:
    import scipy
    print("✓ SciPy:", scipy.__version__)
except:
    print("✗ SciPy failed")
EOF
```

---

## STEP 8: Create Minimal Inference Script

Since you might not have all files yet, create a minimal test:

```bash
cd ~
cat > test_model.py << 'EOF'
#!/usr/bin/env python3
import numpy as np
import onnxruntime as ort
import sys
import os

print("Testing ONNX model on Jetson Xavier NX")
print("=" * 60)

# Check if output.onnx exists
if not os.path.exists('output.onnx'):
    print("Error: output.onnx not found in current directory")
    print("Current directory:", os.getcwd())
    print("Files:", os.listdir('.'))
    sys.exit(1)

# Load model
print("\nLoading ONNX model...")
session = ort.InferenceSession('output.onnx')

# Get model info
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
output_shape = session.get_outputs()[0].shape

print(f"✓ Model loaded successfully")
print(f"  Input: {input_name}, shape: {input_shape}")
print(f"  Output shape: {output_shape}")

# Check providers
providers = session.get_providers()
print(f"\nExecution providers: {providers}")
if 'CUDAExecutionProvider' in providers:
    print("✓ CUDA acceleration enabled!")
else:
    print("⚠ Running on CPU only")

# Create test data
print("\nRunning test inference...")
test_data = np.random.randn(1, 64, 480, 1).astype(np.float32)

# Run inference
import time
start = time.time()
output = session.run(None, {input_name: test_data})[0]
elapsed = (time.time() - start) * 1000

print(f"✓ Inference successful")
print(f"  Output shape: {output.shape}")
print(f"  Inference time: {elapsed:.2f} ms")
print(f"  Predicted class: {np.argmax(output[0])}")
print(f"  Confidence: {output[0][np.argmax(output[0])]:.2%}")

print("\n" + "=" * 60)
print("SUCCESS! Model is working on Jetson.")
print("=" * 60)
EOF

chmod +x test_model.py
```

---

## STEP 9: Test with Your Model

```bash
# Make sure output.onnx is in current directory
ls -lh output.onnx

# Run test
python3 test_model.py
```

**Expected output:**
```
Testing ONNX model on Jetson Xavier NX
============================================================

Loading ONNX model...
✓ Model loaded successfully
  Input: input, shape: [1, 64, 480, 1]
  Output shape: [1, 4]

Execution providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
✓ CUDA acceleration enabled!

Running test inference...
✓ Inference successful
  Output shape: (1, 4)
  Inference time: 2.45 ms
  Predicted class: 1
  Confidence: 87.34%

============================================================
SUCCESS! Model is working on Jetson.
============================================================
```

---

## STEP 10: Enable Maximum Performance

```bash
# Check current power mode
sudo nvpmodel -q

# Set to maximum performance (MAXN - Mode 0)
sudo nvpmodel -m 0

# Lock clocks to maximum
sudo jetson_clocks

# Verify
sudo jetson_clocks --show
```

---

## STEP 11: Monitor Performance

```bash
# In one terminal, monitor system:
tegrastats

# In another terminal, run inference:
python3 test_model.py
```

**What to look for in tegrastats:**
- `GR3D_FREQ` - GPU frequency (should be at max)
- `RAM` - Memory usage
- `CPU` - CPU usage per core
- `TEMP` - Temperature (should stay under 80°C)

---

## 🎯 Quick Benchmark Script

```bash
cat > benchmark.py << 'EOF'
#!/usr/bin/env python3
import numpy as np
import onnxruntime as ort
import time

print("Jetson Xavier NX - Model Benchmark")
print("=" * 60)

session = ort.InferenceSession('output.onnx')
input_name = session.get_inputs()[0].name
test_data = np.random.randn(1, 64, 480, 1).astype(np.float32)

# Warmup
for _ in range(10):
    session.run(None, {input_name: test_data})

# Benchmark
times = []
iterations = 100
print(f"\nRunning {iterations} iterations...")

for i in range(iterations):
    start = time.time()
    session.run(None, {input_name: test_data})
    times.append((time.time() - start) * 1000)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{iterations}")

times = np.array(times)
print("\n" + "=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)
print(f"Mean: {np.mean(times):.2f} ms")
print(f"Std:  {np.std(times):.2f} ms")
print(f"Min:  {np.min(times):.2f} ms")
print(f"Max:  {np.max(times):.2f} ms")
print(f"Median: {np.median(times):.2f} ms")
print(f"Throughput: {1000/np.mean(times):.2f} inferences/sec")
print("=" * 60)
EOF

chmod +x benchmark.py
python3 benchmark.py
```

---

## 📊 Troubleshooting (Ubuntu on Jetson)

### Issue: "Could not load dynamic library 'libcudart.so'"

```bash
# Add CUDA to library path
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Issue: "ImportError: No module named 'tensorflow'"

```bash
# Check installation
pip3 list | grep tensorflow

# If not found, reinstall
pip3 install --user --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow==2.7.0+nv22.1
```

### Issue: Out of disk space

```bash
# Check disk usage
df -h

# Clean up
sudo apt-get clean
sudo apt-get autoremove
rm -rf ~/.cache/pip
```

### Issue: Thermal throttling (slow performance)

```bash
# Check temperature
tegrastats | grep temp

# If over 75°C:
# - Add fan or heatsink
# - Check airflow
# - Reduce clock speed if needed
```

---

## 🎯 Performance Targets

After setup, you should see:

| Metric | Target | Command to Check |
|--------|--------|------------------|
| Inference time | 2-5 ms | `python3 test_model.py` |
| Throughput | 200-400 FPS | `python3 benchmark.py` |
| GPU usage | 30-70% | `tegrastats` |
| Temperature | < 70°C | `tegrastats` |

---

## 📝 Summary Commands (Copy/Paste Ready)

```bash
# Full setup in one go (run on Jetson):
cd ~
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-dev build-essential libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev libjpeg8-dev liblapack-dev libblas-dev gfortran
python3 -m pip install --upgrade pip
pip3 install --user numpy==1.19.4 scipy==1.7.3 scikit-learn==1.0.2
pip3 install --user --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow==2.7.0+nv22.1
pip3 install --user keras==2.6.0
pip3 install --user onnxruntime==1.12.1
sudo nvpmodel -m 0
sudo jetson_clocks
```

Then test:
```bash
python3 test_model.py
python3 benchmark.py
```

---

## Next: Get Your Files

You need to transfer from development machine:
- `output.onnx` (16 KB) - Your trained model
- `eeg_reduction.py` - Preprocessing functions
- `models.py` - Model architecture (if using .h5)
- `jetson_inference.py` - Full inference script

Let me know if you need help with the transfer!
