# Deployment Summary - EEGNet BCI for Jetson Xavier NX

## Overview

This repository has been analyzed and prepared for deployment on NVIDIA Jetson Xavier NX. The deployment package includes all necessary files, scripts, and documentation for running EEGNet-based Brain-Computer Interface inference on the Jetson platform.

## Repository Analysis

### Original Files
- **`main_global.py`** - Global model training script (not needed for inference)
- **`main_ss.py`** - Subject-specific transfer learning script (not needed for inference)
- **`models.py`** - EEGNet model definition (required)
- **`eeg_reduction.py`** - Data preprocessing functions (required)
- **`dependency.yml`** - Conda environment file (reference only)
- **`output.onnx`** - ONNX model file (optional, for ONNX Runtime)
- **`train_val_global_ss.ipynb`** - Training notebook (not needed for inference)

### New Deployment Files Created

1. **`inference.py`** - Main inference script optimized for Jetson
   - Batch and real-time inference modes
   - GPU memory optimization
   - Benchmark capabilities
   - Command-line interface

2. **`realtime_inference.py`** - Real-time inference helper
   - Continuous processing with buffering
   - Thread-based architecture
   - Callback support
   - Statistics tracking

3. **`deploy_jetson.sh`** - Automated deployment script
   - System dependency installation
   - TensorFlow setup for Jetson
   - Python package installation
   - Performance mode configuration

4. **`test_deployment.py`** - Deployment verification script
   - Dependency checking
   - GPU availability testing
   - Model loading verification
   - Inference testing

5. **`requirements_jetson.txt`** - Python dependencies for Jetson

6. **`DEPLOYMENT_JETSON.md`** - Comprehensive deployment guide
   - Step-by-step instructions
   - Troubleshooting guide
   - Performance optimization tips
   - Real-time deployment examples

7. **`README_DEPLOYMENT.md`** - Quick reference guide

## Deployment Architecture

### Inference Pipeline
```
EEG Data → Preprocessing (eeg_reduction) → Model Inference → Predictions
```

### Key Components

1. **Data Preprocessing** (`eeg_reduction.py`)
   - Channel selection (8, 19, 27, 38, or 64 channels)
   - Downsampling (optional)
   - Time window extraction (1, 2, or 3 seconds)

2. **Model Inference** (`inference.py`)
   - Keras/TensorFlow model loading
   - GPU-accelerated inference
   - Batch processing support
   - Real-time single-trial processing

3. **Real-Time Processing** (`realtime_inference.py`)
   - Sample buffering
   - Continuous inference loop
   - Thread-safe operations

## Deployment Steps

### Quick Deployment (Automated)
```bash
# 1. Transfer files to Jetson
scp -r /workspace/* jetson@<jetson-ip>:~/eegnet-deploy/

# 2. Run deployment script
ssh jetson@<jetson-ip>
cd ~/eegnet-deploy
./deploy_jetson.sh

# 3. Copy model files
cp /path/to/models/*.h5 ~/eegnet-bci/models/

# 4. Test deployment
python3 test_deployment.py
```

### Manual Deployment
See `DEPLOYMENT_JETSON.md` for detailed manual setup instructions.

## System Requirements

### Hardware
- NVIDIA Jetson Xavier NX
- 8GB RAM
- MicroSD card (64GB+)
- Power supply (5V/4A)

### Software
- JetPack 4.6+ (TensorFlow 1.x) or JetPack 5.x (TensorFlow 2.x)
- Python 3.6+
- CUDA-enabled GPU drivers

### Dependencies
- TensorFlow (Jetson-optimized version)
- Keras 2.2.4
- NumPy 1.18.5
- SciPy 1.4.1
- scikit-learn 0.22.1
- h5py 2.10.0

## Model Requirements

### Expected Model Format
- **Format**: Keras `.h5` files
- **Input shape**: `(batch, 64, 480, 1)` for 64 channels, 3 seconds at 160 Hz
- **Output shape**: `(batch, num_classes)` with softmax probabilities
- **Classes**: Typically 4 classes (motor imagery tasks)

### Model Files Location
After training, copy model files to:
```
~/eegnet-bci/models/
├── global_class_4_ds1_nch64_T3_split_0.h5
├── global_class_4_ds1_nch64_T3_split_1.h5
└── ...
```

## Usage Examples

### Basic Inference
```bash
python3 inference.py \
    --model models/global_class_4_ds1_nch64_T3_split_0.h5 \
    --data eeg_data.npy
```

### Real-Time Inference
```bash
python3 inference.py \
    --model models/global_class_4_ds1_nch64_T3_split_0.h5 \
    --data eeg_data.npy \
    --realtime
```

### Python API
```python
from inference import EEGNetInference

engine = EEGNetInference('models/your_model.h5')
result = engine.predict_single(eeg_data)
print(f"Class: {result['class']}, Confidence: {result['confidence']}")
```

## Performance Optimization

### Jetson Configuration
```bash
# Maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Recommended Settings
- **Batch size**: 8-16 for inference
- **GPU memory**: Limited to 50% (configured in inference.py)
- **Processing mode**: Real-time for single trials, batch for multiple samples

### Expected Performance
- **Inference time**: ~10-50ms per sample (depending on model size)
- **Throughput**: ~20-100 samples/second (batch processing)
- **Memory usage**: ~1-2GB GPU memory

## Known Issues and Solutions

### BatchNormalization Error
**Issue**: `TypeError: 'NoneType' object is not callable`

**Solution**: Modify TensorFlow backend file (see `DEPLOYMENT_JETSON.md`)

### Out of Memory
**Solution**: 
- Reduce batch size
- Close other applications
- Use model quantization

### Slow Inference
**Solution**:
- Enable performance mode
- Verify GPU is being used
- Optimize batch size

## File Structure

```
/workspace/
├── Core Files (Original)
│   ├── models.py
│   ├── eeg_reduction.py
│   └── output.onnx
│
├── Deployment Files (New)
│   ├── inference.py
│   ├── realtime_inference.py
│   ├── deploy_jetson.sh
│   ├── test_deployment.py
│   └── requirements_jetson.txt
│
└── Documentation (New)
    ├── DEPLOYMENT_JETSON.md
    ├── README_DEPLOYMENT.md
    └── DEPLOYMENT_SUMMARY.md (this file)
```

## Next Steps

1. ✅ **Deployment package created** - All files ready
2. ⏭️ **Transfer to Jetson** - Copy files to target device
3. ⏭️ **Run deployment script** - Automated setup
4. ⏭️ **Copy model files** - Add trained models
5. ⏭️ **Test deployment** - Verify everything works
6. ⏭️ **Integrate with EEG hardware** - Connect acquisition device
7. ⏭️ **Optimize for real-time** - Fine-tune performance

## Additional Notes

### Missing Files
- **`get_data.py`** - Referenced in training scripts but not present
  - **Impact**: Not needed for inference-only deployment
  - **Action**: None required

### Model Training
- Training scripts (`main_global.py`, `main_ss.py`) are not needed for deployment
- Models should be trained on a development machine with GPU
- Transfer only the trained `.h5` model files to Jetson

### ONNX Model
- `output.onnx` file is present but not used in current deployment
- Can be used with ONNX Runtime for alternative inference (requires additional setup)

## Support and Documentation

- **Quick Start**: See `README_DEPLOYMENT.md`
- **Detailed Guide**: See `DEPLOYMENT_JETSON.md`
- **Testing**: Run `test_deployment.py`
- **Troubleshooting**: See troubleshooting section in `DEPLOYMENT_JETSON.md`

## License

Please refer to the LICENSE file for licensing information.

---

**Deployment Package Status**: ✅ Complete and Ready

All necessary files have been created and are ready for deployment on Jetson Xavier NX.
