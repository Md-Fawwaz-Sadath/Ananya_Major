# Jetson Xavier NX Storage Cleanup Guide

## 🧹 Complete Storage Cleanup for Jetson Xavier NX

### Quick Storage Check

First, let's see what's using space:

```bash
# Check overall disk usage
df -h

# Check what's taking up space
du -h --max-depth=1 /home | sort -hr
du -h --max-depth=1 /usr | sort -hr
du -h --max-depth=1 /var | sort -hr

# Find largest files
sudo find / -type f -size +100M -exec ls -lh {} \; 2>/dev/null | sort -k5 -hr | head -20
```

---

## 🚨 Nuclear Option: Complete Cleanup (Most Aggressive)

### Method 1: Clean Everything Non-Essential

```bash
# WARNING: This removes A LOT. Read each section before running!

# 1. Clean APT cache (safe)
sudo apt clean
sudo apt autoclean
sudo apt autoremove -y

# 2. Remove old kernels (keeps current one)
sudo apt autoremove --purge -y

# 3. Clean pip cache (safe)
pip3 cache purge
rm -rf ~/.cache/pip

# 4. Remove Python bytecode and caches
find ~/ -type f -name '*.pyc' -delete
find ~/ -type d -name '__pycache__' -delete
find ~/ -type d -name '.pytest_cache' -delete

# 5. Clean system logs (safe)
sudo journalctl --vacuum-size=50M
sudo rm -rf /var/log/*.gz
sudo rm -rf /var/log/*.1
sudo rm -rf /var/log/*.old

# 6. Remove temporary files
sudo rm -rf /tmp/*
sudo rm -rf /var/tmp/*

# 7. Clean thumbnail cache
rm -rf ~/.cache/thumbnails/*

# 8. Remove documentation (if you don't need it)
sudo rm -rf /usr/share/doc/*
sudo rm -rf /usr/share/man/*

# 9. Remove sample videos/images (if present)
rm -rf ~/Videos/*
rm -rf ~/Pictures/sample*

# Check space gained
df -h
```

---

## 🎯 Targeted Cleanup (Recommended Approach)

### Step 1: Remove Docker (if installed and not needed)

```bash
# Docker can take 5-10+ GB
docker system prune -a --volumes  # If you use docker
# Or completely remove Docker
sudo apt purge docker-ce docker-ce-cli containerd.io -y
sudo rm -rf /var/lib/docker
sudo rm -rf /var/lib/containerd
```

### Step 2: Remove Unnecessary Development Tools

```bash
# Remove if you don't need them (saves 2-5 GB)
sudo apt remove --purge libreoffice* -y
sudo apt remove --purge thunderbird* -y
sudo apt remove --purge firefox -y

# Remove build tools if you won't compile anything
sudo apt remove --purge build-essential -y
sudo apt remove --purge gcc g++ -y

# Remove if you don't use GUI
sudo apt remove --purge chromium-browser -y
```

### Step 3: Clean Conda/Miniconda (if installed)

```bash
# Conda can be HUGE (10+ GB)
conda clean --all -y

# Or remove entirely if not needed
rm -rf ~/miniconda3
rm -rf ~/anaconda3
rm -rf ~/.conda
```

### Step 4: Remove TensorFlow/PyTorch if Not Needed

```bash
# If you're using TensorRT only, you don't need full TensorFlow
pip3 uninstall tensorflow tensorflow-gpu -y
pip3 uninstall torch torchvision torchaudio -y

# This can save 1-3 GB
```

### Step 5: Clean CUDA Samples and Documentation

```bash
# CUDA samples can take 1-2 GB
sudo rm -rf /usr/local/cuda/samples
sudo rm -rf /usr/local/cuda/doc
sudo rm -rf /usr/local/cuda/extras
```

### Step 6: Remove Old Datasets

```bash
# If you have datasets you don't need anymore
rm -rf ~/datasets/*
rm -rf ~/data/*
rm -rf ~/.keras/datasets/*
```

---

## 🔧 Advanced Cleanup Commands

### Find and Remove Large Files

```bash
# Find files larger than 500MB
sudo find / -type f -size +500M -exec ls -lh {} \; 2>/dev/null

# Example: Remove specific large file
# sudo rm /path/to/large/file
```

### Clean Specific Directories

```bash
# Clean pip packages you don't need
pip3 list | tail -n +3 | awk '{print $1}' | grep -v "pip\|setuptools\|wheel"
# Then manually uninstall what you don't need:
# pip3 uninstall <package-name> -y

# Remove unused Python versions (if multiple installed)
sudo apt remove python2.7 python3.6 -y  # Keep only what you need
```

### Remove Swap File (if you have one and don't need it)

```bash
# Check if swap exists
sudo swapon --show

# Disable and remove swap file (ONLY if you have enough RAM)
sudo swapoff -a
sudo rm /swapfile  # or wherever it is
# Edit /etc/fstab to remove swap entry
```

---

## 🎮 Minimal Setup for EEGNet Deployment

After cleanup, here's what you actually need:

### Essential System Components (DO NOT REMOVE)
- ✅ JetPack base system
- ✅ TensorRT (pre-installed)
- ✅ CUDA runtime (not samples)
- ✅ Python 3

### Essential Python Packages (Only These)
```bash
pip3 install numpy scipy pycuda
```

### Your EEGNet Files (Total ~1-2 MB)
```
output.onnx              (0.02 MB)
model.trt               (will be ~0.1-1 MB)
onnx_to_tensorrt.py     (9 KB)
run_eeg.py              (14 KB)
models.py               (4.3 KB)
eeg_reduction.py        (2.8 KB)
test_data.npy           (5-10 MB for testing)
```

**Total space needed: Less than 100 MB for your project!**

---

## 📊 Complete Cleanup Script

Save this as `cleanup_jetson.sh`:

```bash
#!/bin/bash

echo "========================================================================"
echo "Jetson Xavier NX Complete Cleanup Script"
echo "========================================================================"
echo ""
echo "WARNING: This will remove many files and packages."
echo "Review the script before running!"
echo ""
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cleanup cancelled"
    exit 0
fi

echo ""
echo "Starting cleanup..."
echo ""

# Safe cleanups
echo "[1/10] Cleaning APT cache..."
sudo apt clean
sudo apt autoclean
sudo apt autoremove -y

echo "[2/10] Cleaning pip cache..."
pip3 cache purge
rm -rf ~/.cache/pip

echo "[3/10] Removing Python bytecode..."
find ~/ -type f -name '*.pyc' -delete 2>/dev/null
find ~/ -type d -name '__pycache__' -delete 2>/dev/null

echo "[4/10] Cleaning system logs..."
sudo journalctl --vacuum-size=50M
sudo rm -rf /var/log/*.gz 2>/dev/null
sudo rm -rf /var/log/*.1 2>/dev/null
sudo rm -rf /var/log/*.old 2>/dev/null

echo "[5/10] Removing temporary files..."
sudo rm -rf /tmp/* 2>/dev/null
sudo rm -rf /var/tmp/* 2>/dev/null

echo "[6/10] Cleaning thumbnail cache..."
rm -rf ~/.cache/thumbnails/* 2>/dev/null
rm -rf ~/.thumbnails/* 2>/dev/null

echo "[7/10] Removing documentation..."
sudo rm -rf /usr/share/doc/* 2>/dev/null
sudo rm -rf /usr/share/man/* 2>/dev/null

echo "[8/10] Cleaning CUDA samples..."
sudo rm -rf /usr/local/cuda/samples 2>/dev/null
sudo rm -rf /usr/local/cuda/doc 2>/dev/null

echo "[9/10] Removing sample media..."
rm -rf ~/Videos/* 2>/dev/null
rm -rf ~/Pictures/sample* 2>/dev/null

echo "[10/10] Final cleanup..."
sudo apt autoremove --purge -y
sudo apt autoclean

echo ""
echo "========================================================================"
echo "Cleanup Complete!"
echo "========================================================================"
echo ""
echo "Disk usage after cleanup:"
df -h /
echo ""
echo "Space in home directory:"
du -sh ~
echo ""
echo "========================================================================"
```

---

## 🚀 Extreme Nuclear Option: Flash Fresh

If you need maximum space and minimal installation:

### Option 1: Minimal JetPack Installation

When flashing JetPack, choose **minimal installation**:
- Only CUDA runtime (not samples)
- No GUI desktop environment
- No development tools
- Just terminal access

This can give you **10-15 GB free** vs full installation.

### Option 2: Command-Line Only Setup

```bash
# Remove GUI entirely (saves 3-5 GB)
sudo systemctl set-default multi-user.target
sudo apt remove --purge ubuntu-desktop -y
sudo apt remove --purge gdm3 -y
sudo apt autoremove -y
```

**Note:** After this, you'll only have command-line access (SSH or serial).

---

## 📦 Storage Recommendations

### Minimum Storage Needed for EEGNet Deployment

| Component | Size |
|-----------|------|
| JetPack Base | 3-5 GB |
| Python + NumPy/SciPy | 500 MB |
| TensorRT (pre-installed) | 0 MB (included) |
| Your EEGNet project | < 100 MB |
| Test data | 10-50 MB |
| **Total** | **~4-6 GB** |

### After Cleanup You Should Have

- **Minimal setup:** 10-15 GB free
- **With cleanup script:** 8-12 GB free
- **With aggressive cleanup:** 15-20 GB free

---

## 🔍 Before/After Check

```bash
# Before cleanup
echo "=== BEFORE CLEANUP ===" > cleanup_report.txt
df -h >> cleanup_report.txt
du -sh ~ >> cleanup_report.txt
echo "" >> cleanup_report.txt

# Run cleanup
./cleanup_jetson.sh

# After cleanup
echo "=== AFTER CLEANUP ===" >> cleanup_report.txt
df -h >> cleanup_report.txt
du -sh ~ >> cleanup_report.txt

# Compare
cat cleanup_report.txt
```

---

## ⚠️ Important Warnings

### DO NOT REMOVE
- ❌ `/usr/lib/aarch64-linux-gnu/` (system libraries)
- ❌ `/usr/lib/aarch64-linux-gnu/tegra/` (Tegra libraries)
- ❌ `/usr/local/cuda/lib64/` (CUDA libraries)
- ❌ TensorRT libraries
- ❌ System Python packages needed by OS

### Safe to Remove (After Verification)
- ✅ Old Docker images
- ✅ Conda/Anaconda
- ✅ Old datasets
- ✅ CUDA samples
- ✅ Documentation
- ✅ GUI applications (if using headless)
- ✅ Old logs

---

## 🎯 Quick Command Reference

```bash
# Check disk usage
df -h

# Find biggest directories
sudo du -h --max-depth=1 / 2>/dev/null | sort -hr | head -20

# Find biggest files
sudo find / -type f -size +100M 2>/dev/null | xargs ls -lh | sort -k5 -hr

# Clean everything safely
sudo apt clean && sudo apt autoremove -y && pip3 cache purge

# Check what packages are installed
dpkg --get-selections | grep -v deinstall | wc -l

# See largest installed packages
dpkg-query -Wf '${Installed-Size}\t${Package}\n' | sort -n | tail -20
```

---

## 📞 Need More Space?

If you still don't have enough space after cleanup:

1. **Use USB/SSD storage** - Mount external drive for datasets/models
2. **Network storage** - Keep data on network drive, only load what's needed
3. **Flash minimal JetPack** - Start fresh with minimal installation
4. **Larger SD card** - Upgrade to 128GB or 256GB card

---

## ✅ Recommended Cleanup Order

1. ✅ Run the cleanup script above
2. ✅ Remove Docker if installed
3. ✅ Remove Conda if installed
4. ✅ Remove GUI if not needed
5. ✅ Remove old datasets
6. ✅ Remove CUDA samples
7. ✅ Remove unused Python packages

After this, you should have **10-15 GB free** which is more than enough for EEGNet deployment!

---

**Save this guide and run the cleanup script on your Jetson! 🧹**
