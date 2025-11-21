# 🚀 NEXT STEPS: Deploy to Jetson Xavier NX

## Current Situation
✅ You are on your **development machine** (workspace)
✅ Deployment package is ready: `eegnet_jetson_deployment_20251121_070734.tar.gz`
⏳ Need to transfer to **Jetson Xavier NX** and run setup

---

## PHASE 1: Prepare Your Jetson Xavier NX
**⚠️ Do this BEFORE transferring files**

### Prerequisites Checklist
- [ ] Jetson Xavier NX is powered on and booted
- [ ] JetPack is installed (4.6.x or 5.x recommended)
- [ ] Jetson is connected to network (WiFi or Ethernet)
- [ ] You know the Jetson's IP address
- [ ] You can SSH to Jetson (or have monitor/keyboard connected)

### How to Find Your Jetson's IP Address

**Option A - On the Jetson itself (with monitor/keyboard):**
```bash
# On Jetson, open terminal and run:
hostname -I
# Example output: 192.168.1.100
```

**Option B - From your router:**
- Check your router's admin page for connected devices
- Look for device named "jetson" or similar

**Option C - Scan your network (from current machine):**
```bash
# Install nmap if needed
sudo apt-get install nmap

# Scan your network (replace with your subnet)
nmap -sn 192.168.1.0/24 | grep -B 2 jetson
```

---

## PHASE 2: Transfer Files to Jetson
**🖥️ Run these commands on YOUR CURRENT MACHINE (not Jetson)**

### Find the Deployment Package
```bash
# Check the package exists
cd /workspace
ls -lh eegnet_jetson_deployment_*.tar.gz

# Expected output:
# eegnet_jetson_deployment_20251121_070734.tar.gz
```

### Option A: Transfer via SCP (Recommended if Jetson is on network)

```bash
# Replace <jetson-ip> with your Jetson's actual IP address
# Replace 'jetson' with your Jetson username if different

# 1. Transfer the package
scp eegnet_jetson_deployment_*.tar.gz jetson@<jetson-ip>:~/

# Example:
# scp eegnet_jetson_deployment_*.tar.gz jetson@192.168.1.100:~/

# You'll be prompted for the Jetson's password
# Default is usually the password you set during JetPack setup
```

**Troubleshooting SCP:**
```bash
# If you get "connection refused":
# 1. Check Jetson is on and network connected
# 2. Verify IP address: ping <jetson-ip>
# 3. Check SSH is enabled on Jetson

# If you get "permission denied":
# - Verify username (default is usually 'jetson' or 'nvidia')
# - Verify password
```

### Option B: Transfer via USB Drive

If you don't have network connectivity:

```bash
# 1. Copy to USB drive (on current machine)
cp eegnet_jetson_deployment_*.tar.gz /media/your-usb-drive/

# 2. Unmount USB safely
sync
sudo umount /media/your-usb-drive

# 3. Move USB to Jetson
# 4. On Jetson, mount and copy (see PHASE 3)
```

---

## PHASE 3: Connect to Jetson and Setup
**🤖 Now switch to JETSON - run these commands ON THE JETSON**

### Connect to Jetson

**Option A - SSH (Remote):**
```bash
# From your current machine, SSH to Jetson
ssh jetson@<jetson-ip>

# Example:
# ssh jetson@192.168.1.100

# Enter password when prompted
# You are now ON the Jetson terminal
```

**Option B - Direct Access (Monitor + Keyboard):**
- Connect monitor and keyboard to Jetson
- Open Terminal application
- You're now ready to run commands

### Extract Deployment Package

```bash
# You should now be in a Jetson terminal
# Verify you're on Jetson:
uname -a
# Should show something like "aarch64" and "tegra"

# List transferred file
ls -lh ~/*.tar.gz

# Extract the package
cd ~
tar -xzf eegnet_jetson_deployment_*.tar.gz

# Enter the directory
cd eegnet_jetson_deployment_*/

# List contents to verify
ls -la
```

**Expected output:**
```
docs/
models/
eeg_reduction.py
jetson_inference.py
jetson_setup.sh
models.py
QUICKSTART.txt
requirements_jetson.txt
transfer_to_jetson.sh
LICENSE
```

---

## PHASE 4: Run Setup on Jetson
**🤖 Still ON THE JETSON - this takes 30-60 minutes**

### Before Running Setup

Check your current Python and system:
```bash
# Check Python version
python3 --version

# Check available disk space (need ~2-3 GB)
df -h

# Check JetPack version
cat /etc/nv_tegra_release
```

### Run the Setup Script

```bash
# Make sure you're in the deployment directory
cd ~/eegnet_jetson_deployment_*/

# Make script executable
chmod +x jetson_setup.sh

# Run setup (this will take 30-60 minutes)
./jetson_setup.sh

# Follow prompts:
# - It will ask for sudo password for system packages
# - It will ask if you want to enable max performance mode (say 'y')
```

**What the script does:**
1. ✅ Updates system packages
2. ✅ Installs system dependencies (HDF5, BLAS, etc.)
3. ✅ Installs TensorFlow for Jetson
4. ✅ Installs Keras
5. ✅ Installs ONNX Runtime
6. ✅ Installs Python packages (NumPy, SciPy, etc.)
7. ✅ Verifies installation
8. ✅ Enables max performance mode (optional)

**During installation:**
- Don't close the terminal
- It's normal to see lots of compilation output
- May see some warnings (usually okay)
- Should not see red ERROR messages

### If Setup Fails

Save the error output and share it with me:
```bash
# Re-run with output saved to file
./jetson_setup.sh 2>&1 | tee setup_log.txt

# If it fails, you can show me the log
cat setup_log.txt
```

---

## PHASE 5: Test Inference on Jetson
**🤖 Still ON THE JETSON - testing takes 1-2 minutes**

### Verify Installation

```bash
# Still in ~/eegnet_jetson_deployment_*/

# Quick verification
python3 -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python3 -c "import onnxruntime; print('ONNX Runtime:', onnxruntime.__version__)"
python3 -c "import numpy; print('NumPy:', numpy.__version__)"
```

### Run Demo Inference

```bash
# Test with ONNX model (should be fast)
python3 jetson_inference.py --model models/output.onnx --mode demo --num-samples 5

# This will:
# 1. Load the ONNX model
# 2. Generate 5 synthetic EEG samples
# 3. Run inference
# 4. Show predictions and timing
```

**Expected output:**
```
================================================================
  EEGNet Jetson Xavier NX Inference Demo
================================================================

✓ CUDA acceleration enabled for ONNX Runtime
Model loaded: models/output.onnx
Input shape: [1, 64, 480, 1]
Output shape: [1, 4]

Generating 5 synthetic EEG trials...

Running inference...

================================================================
INFERENCE RESULTS
================================================================

Trial 1:
  Predicted: Left Hand
  Confidence: 87.34%
  Inference time: 2.45 ms

Trial 2:
  Predicted: Right Hand
  Confidence: 92.11%
  Inference time: 2.38 ms

...

================================================================
Average inference time: 2.42 ms
Throughput: 413.22 FPS
================================================================
```

### Run Benchmark

```bash
# Benchmark for 100 iterations
python3 jetson_inference.py --model models/output.onnx --mode benchmark --num-iterations 100
```

**Expected output:**
```
================================================================
BENCHMARK RESULTS
================================================================
Mean inference time: 2.50 ms
Std deviation: 0.15 ms
Min: 2.30 ms
Max: 3.20 ms
Throughput: 400.00 inferences/sec
================================================================
```

---

## PHASE 6: Enable Maximum Performance (Optional)
**🤖 ON THE JETSON - for best inference speed**

```bash
# Check current power mode
sudo nvpmodel -q

# Set to maximum performance (MAXN mode)
sudo nvpmodel -m 0

# Lock clocks to maximum
sudo jetson_clocks

# Verify
sudo jetson_clocks --show
```

**Then re-run benchmark to see improved performance:**
```bash
python3 jetson_inference.py --model models/output.onnx --mode benchmark --num-iterations 100
```

---

## 📊 PHASE 7: Monitor Performance
**🤖 ON THE JETSON**

### Real-time Monitoring

```bash
# Open another terminal on Jetson and run:
tegrastats

# This shows:
# - GPU utilization (GR3D)
# - CPU usage per core
# - Memory usage
# - Temperature
# - Power consumption
```

### While monitoring, run inference in first terminal:
```bash
python3 jetson_inference.py --model models/output.onnx --mode benchmark --num-iterations 1000
```

Watch `tegrastats` to see GPU activity!

---

## 🔗 "How to Link Jetson to You (AI Assistant)"

**Important:** I (the AI) cannot directly connect to your Jetson. However, you can share information with me:

### What You Can Share With Me:

1. **Error messages:**
```bash
# On Jetson, if something fails:
./jetson_setup.sh 2>&1 | tee error_log.txt
# Then copy/paste the error_log.txt content to me
```

2. **System information:**
```bash
# Gather system info
cat /etc/nv_tegra_release
python3 --version
nvidia-smi  # or tegrastats output
df -h
free -h

# Copy and paste output to me for troubleshooting
```

3. **Inference results:**
```bash
# Run inference and copy the output
python3 jetson_inference.py --model models/output.onnx --mode demo > results.txt
cat results.txt
# Share results.txt with me
```

4. **Performance metrics:**
```bash
# Run benchmark and share results
python3 jetson_inference.py --model models/output.onnx --mode benchmark > benchmark.txt
cat benchmark.txt
```

### If You Need Help:

Just paste the command output, error messages, or logs in our conversation, and I'll help you troubleshoot!

---

## 📝 Quick Command Summary

### On YOUR CURRENT MACHINE (workspace):
```bash
# 1. Transfer package to Jetson
cd /workspace
scp eegnet_jetson_deployment_*.tar.gz jetson@<jetson-ip>:~/
```

### On JETSON XAVIER NX:
```bash
# 2. Extract package
cd ~
tar -xzf eegnet_jetson_deployment_*.tar.gz
cd eegnet_jetson_deployment_*/

# 3. Run setup (30-60 min)
chmod +x jetson_setup.sh
./jetson_setup.sh

# 4. Test inference
python3 jetson_inference.py --model models/output.onnx --mode demo

# 5. Benchmark
python3 jetson_inference.py --model models/output.onnx --mode benchmark

# 6. Enable max performance
sudo nvpmodel -m 0
sudo jetson_clocks
```

---

## ✅ Success Criteria

You'll know it's working when you see:
- ✅ Setup script completes without errors
- ✅ TensorFlow and ONNX Runtime import successfully
- ✅ Demo inference runs and shows predictions
- ✅ Inference time is 2-5 ms (ONNX + CUDA)
- ✅ Throughput is 200-400 FPS

---

## 🆘 Troubleshooting

### Common Issues

**"Connection refused" when using SCP:**
- Verify Jetson IP: `ping <jetson-ip>`
- Check Jetson is on network
- Try direct connection with monitor/keyboard + USB drive

**"Out of disk space" during setup:**
- Check: `df -h`
- Need at least 3GB free
- Clean up: `sudo apt-get clean`

**"CUDA not available" after setup:**
- Check: `python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- Verify JetPack includes CUDA
- May need to restart Jetson

**Slow inference (>10ms):**
- Enable max performance: `sudo nvpmodel -m 0 && sudo jetson_clocks`
- Check temperature: `tegrastats | grep temp`
- Add cooling if overheating

### Get Help from AI

Share any of these with me:
```bash
# System info
uname -a
cat /etc/nv_tegra_release

# Error logs
cat setup_log.txt

# Python package versions
pip3 list | grep -E "tensorflow|keras|onnx|numpy"

# Inference output
python3 jetson_inference.py --model models/output.onnx --mode demo 2>&1
```

---

## 🎯 Next Steps After Successful Deployment

Once everything works:

1. **Integrate Real EEG Hardware**
   - Connect your EEG device to Jetson (USB/Serial)
   - Stream data in real-time
   - Use the preprocessing pipeline in `eeg_reduction.py`

2. **Build Your Application**
   - Use `jetson_inference.py` as a library
   - Create custom control logic based on predictions
   - Implement real-time feedback

3. **Optimize Further**
   - Convert to TensorRT for 2-3x speedup
   - Implement data buffering
   - Add logging and monitoring

---

**Ready to start? Begin with PHASE 1 above! 🚀**

Share any errors or questions with me and I'll help troubleshoot!
