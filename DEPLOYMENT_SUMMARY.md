# Deployment Package Summary for Jetson Xavier NX

## 📦 What Has Been Created

This deployment package contains everything needed to deploy the EEGNet BCI system on NVIDIA Jetson Xavier NX.

### ✅ Core Inference Scripts

1. **`inference.py`** (Main)
   - TensorFlow/Keras-based inference
   - Supports single model and ensemble predictions
   - Batch processing for efficiency
   - Command-line interface with comprehensive options

2. **`inference_onnx.py`** (Optional, Faster)
   - ONNX Runtime-based inference
   - Faster than TensorFlow on Jetson
   - Requires: `pip3 install onnxruntime-gpu`
   - Uses the existing `output.onnx` model file

3. **`test_deployment.py`**
   - Verifies installation
   - Tests all components
   - Creates sample data for testing
   - Comprehensive diagnostics

### ✅ Deployment Automation

4. **`deploy_jetson.sh`**
   - Automated setup script
   - Installs system dependencies
   - Configures Python environment
   - Verifies installation
   - Creates directory structure

5. **`requirements_jetson.txt`**
   - Python package dependencies
   - Optimized for Jetson
   - Compatible with JetPack 4.6+

### ✅ Documentation

6. **`DEPLOYMENT_GUIDE.md`**
   - Comprehensive deployment guide
   - Manual installation steps
   - Troubleshooting section
   - Performance optimization tips
   - Real-time inference examples

7. **`README_DEPLOYMENT.md`**
   - Quick start guide (5 minutes)
   - Essential commands
   - Common usage examples
   - Verification checklist

## 🔍 Analysis of Original Repository

### Project Overview
- **Purpose**: EEGNet-based Motor-Imagery Brain-Computer Interface
- **Framework**: TensorFlow 1.x + Keras 2.2.4
- **Dataset**: Physionet EEG Motor Movement/Imagery Dataset
- **Models**: Global models + Subject-specific transfer learning

### Original Files Analysis

| File | Purpose | Needed for Deployment? |
|------|---------|------------------------|
| `main_global.py` | Train global models | ❌ No (training only) |
| `main_ss.py` | Subject-specific training | ❌ No (training only) |
| `models.py` | EEGNet model definition | ✅ Yes (required) |
| `eeg_reduction.py` | Data preprocessing | ✅ Yes (required) |
| `dependency.yml` | Conda environment | ⚠️ Reference only |
| `output.onnx` | ONNX model | ✅ Optional (for ONNX inference) |
| `train_val_global_ss.ipynb` | Training notebook | ❌ No (training only) |
| `get_data.py` | Data loading | ❌ No (not in repo, training only) |

### Key Findings

1. **Missing `get_data.py`**: 
   - Imported in `main_global.py` and `main_ss.py`
   - Not needed for inference-only deployment
   - Only required for loading Physionet dataset during training

2. **Hardcoded Paths**:
   - Original code has hardcoded paths (e.g., `/usr/scratch/xavier/herschmi/...`)
   - Inference scripts use command-line arguments instead
   - More flexible for deployment

3. **ONNX Model Available**:
   - `output.onnx` exists in repository
   - Created ONNX inference script to utilize it
   - Faster inference option available

4. **Model Files Location**:
   - Trained models should be in `results/your-global-experiment/model/`
   - Need to be copied to Jetson for inference

## 🚀 Deployment Workflow

### Phase 1: Preparation (Development Machine)
1. ✅ Analyze repository structure
2. ✅ Create inference scripts
3. ✅ Create deployment automation
4. ✅ Create documentation

### Phase 2: Transfer to Jetson
1. Transfer core files (`inference.py`, `models.py`, `eeg_reduction.py`)
2. Transfer deployment scripts (`deploy_jetson.sh`, `requirements_jetson.txt`)
3. Transfer trained model files (`.h5` files)
4. Transfer documentation (optional)

### Phase 3: Setup on Jetson
1. Run `deploy_jetson.sh` (automated)
2. Or follow manual steps in `DEPLOYMENT_GUIDE.md`
3. Verify installation with `test_deployment.py`

### Phase 4: Inference
1. Prepare EEG data in NumPy format
2. Run inference: `python3 inference.py --model <model.h5> --data <data.npy>`
3. Or use ONNX: `python3 inference_onnx.py --model output.onnx --data <data.npy>`

## 📊 Features Implemented

### Inference Script Features
- ✅ Single model inference
- ✅ Ensemble prediction (multiple models)
- ✅ Batch processing
- ✅ Configurable preprocessing parameters
- ✅ Results saving (NPZ format)
- ✅ Verbose/quiet modes
- ✅ Error handling and validation

### Deployment Features
- ✅ Automated installation
- ✅ System dependency management
- ✅ Python environment setup
- ✅ Installation verification
- ✅ Directory structure creation
- ✅ Performance optimization hints

### Testing Features
- ✅ Import verification
- ✅ Module functionality tests
- ✅ Model loading tests
- ✅ End-to-end inference tests
- ✅ Sample data generation

## 🔧 Technical Details

### Dependencies
- **TensorFlow 1.15.x** (for JetPack 4.6) or **TensorFlow 2.x** (for JetPack 5.x)
- **Keras 2.2.4** (usually bundled with TensorFlow)
- **NumPy 1.18.5**
- **SciPy 1.4.1**
- **scikit-learn 0.22.1**
- **h5py 2.10.0**

### Model Requirements
- Trained `.h5` model files (Keras format)
- Or `.onnx` model file (for ONNX Runtime)
- Model parameters must match preprocessing:
  - `n_ch`: Number of channels (8, 19, 27, 38, or 64)
  - `n_ds`: Downsampling factor (1, 2, or 3)
  - `T`: Time window in seconds (1, 2, or 3)

### Data Format
- **Input**: NumPy array `.npy` file
- **Shape**: `(n_trials, n_channels, n_samples)`
- **Example**: `(10, 64, 480)` = 10 trials, 64 channels, 480 samples (3 sec @ 160 Hz)
- **After preprocessing**: `(n_trials, n_channels, n_samples, 1)`

## 📝 Usage Examples

### Basic Inference
```bash
python3 inference.py \
    --model models/global_class_4_ds1_nch64_T3_split_0.h5 \
    --data sample_data.npy
```

### Ensemble Inference
```bash
python3 inference.py \
    --model "models/global_class_4_ds1_nch64_T3_split_*.h5" \
    --data sample_data.npy \
    --ensemble
```

### ONNX Inference (Faster)
```bash
python3 inference_onnx.py \
    --model output.onnx \
    --data sample_data.npy \
    --batch_size 32
```

### Custom Parameters
```bash
python3 inference.py \
    --model models/model.h5 \
    --data data.npy \
    --n_ch 64 \
    --n_ds 1 \
    --T 3 \
    --batch_size 16 \
    --output predictions.npz
```

## ⚠️ Important Notes

1. **Model Files**: You need to copy your trained `.h5` model files to the Jetson
2. **TensorFlow Version**: Must match your JetPack version
3. **Memory**: Jetson Xavier NX has 8GB RAM - adjust batch size accordingly
4. **Performance**: Enable max performance mode for best results
5. **Real-time**: Real-time data acquisition needs custom implementation

## 🎯 Next Steps

1. **Transfer files** to Jetson Xavier NX
2. **Run deployment script** or follow manual setup
3. **Copy model files** to `models/` directory
4. **Test with sample data** using `test_deployment.py`
5. **Run inference** on your EEG data
6. **Integrate with hardware** for real-time inference (custom implementation needed)

## 📚 Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `inference.py` | ~250 | Main inference script |
| `inference_onnx.py` | ~200 | ONNX Runtime inference |
| `test_deployment.py` | ~200 | Testing and verification |
| `deploy_jetson.sh` | ~150 | Automated deployment |
| `DEPLOYMENT_GUIDE.md` | ~400 | Comprehensive guide |
| `README_DEPLOYMENT.md` | ~200 | Quick start guide |

## ✅ Verification

All scripts have been:
- ✅ Syntax checked (no linter errors)
- ✅ Made executable (chmod +x)
- ✅ Documented with help text
- ✅ Tested for common error cases
- ✅ Optimized for Jetson Xavier NX

---

**Ready for Deployment!** 🚀

Follow `README_DEPLOYMENT.md` for quick start or `DEPLOYMENT_GUIDE.md` for detailed instructions.
