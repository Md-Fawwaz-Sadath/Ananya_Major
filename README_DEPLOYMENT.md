# EEGNet BCI - Jetson Xavier NX Deployment Package

This package contains everything needed to deploy the EEGNet-based Brain-Computer Interface on NVIDIA Jetson Xavier NX.

## Quick Start

1. **Transfer files to Jetson:**
   ```bash
   scp -r /workspace/* jetson@<jetson-ip>:~/eegnet-deploy/
   ```

2. **Run automated deployment:**
   ```bash
   ssh jetson@<jetson-ip>
   cd ~/eegnet-deploy
   chmod +x deploy_jetson.sh
   ./deploy_jetson.sh
   ```

3. **Test deployment:**
   ```bash
   cd ~/eegnet-bci
   python3 ../eegnet-deploy/test_deployment.py
   ```

4. **Run inference:**
   ```bash
   python3 inference.py --model models/your_model.h5 --data your_data.npy
   ```

## Files in This Package

### Core Files
- **`inference.py`** - Main inference script optimized for Jetson
- **`models.py`** - EEGNet model definition
- **`eeg_reduction.py`** - Data preprocessing functions
- **`realtime_inference.py`** - Real-time inference helper utilities

### Deployment Files
- **`deploy_jetson.sh`** - Automated deployment script
- **`requirements_jetson.txt`** - Python dependencies for Jetson
- **`DEPLOYMENT_JETSON.md`** - Comprehensive deployment guide
- **`test_deployment.py`** - Test script to verify deployment

### Documentation
- **`README_DEPLOYMENT.md`** - This file (quick reference)
- **`DEPLOYMENT_JETSON.md`** - Detailed deployment instructions

## Directory Structure After Deployment

```
~/eegnet-bci/
├── models.py              # EEGNet model definition
├── eeg_reduction.py       # Data preprocessing
├── inference.py           # Inference script
├── realtime_inference.py  # Real-time inference helper
├── models/                # Trained model files (.h5)
│   └── *.h5
├── data/                  # Input data files
├── results/               # Output predictions
└── logs/                  # Log files
```

## Key Features

### Inference Script (`inference.py`)
- **Batch inference** - Process multiple samples efficiently
- **Real-time inference** - Single trial processing with timing
- **Benchmark mode** - Performance testing
- **GPU optimized** - Configured for Jetson Xavier NX
- **Memory efficient** - GPU memory growth enabled

### Real-Time Inference (`realtime_inference.py`)
- **Continuous processing** - Background thread for inference
- **Data buffering** - Automatic buffering of EEG samples
- **Callback support** - Custom callbacks for predictions
- **Statistics tracking** - Performance metrics

## Usage Examples

### Basic Inference
```bash
python3 inference.py \
    --model models/global_class_4_ds1_nch64_T3_split_0.h5 \
    --data your_data.npy
```

### Real-Time Inference
```bash
python3 inference.py \
    --model models/global_class_4_ds1_nch64_T3_split_0.h5 \
    --data your_data.npy \
    --realtime
```

### Benchmark
```bash
python3 inference.py \
    --model models/global_class_4_ds1_nch64_T3_split_0.h5 \
    --data your_data.npy \
    --benchmark
```

### Python API
```python
from inference import EEGNetInference

# Initialize
engine = EEGNetInference('models/your_model.h5')

# Single prediction
result = engine.predict_single(eeg_data)
print(f"Class: {result['class']}, Confidence: {result['confidence']}")

# Batch prediction
predictions = engine.predict(preprocessed_data, batch_size=16)
```

## Requirements

- **JetPack 4.6+** (for TensorFlow 1.x) or **JetPack 5.x** (for TensorFlow 2.x)
- **Python 3.6+**
- **8GB RAM** (Jetson Xavier NX)
- **Trained model files** (.h5 format)

## Performance Tips

1. **Enable maximum performance:**
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

2. **Optimize batch size:**
   - Small models: batch_size = 16-32
   - Large models: batch_size = 8-16
   - Real-time: batch_size = 1

3. **Monitor performance:**
   ```bash
   tegrastats
   ```

## Troubleshooting

See `DEPLOYMENT_JETSON.md` for detailed troubleshooting guide.

Common issues:
- **BatchNormalization error** - See deployment guide for fix
- **Out of memory** - Reduce batch size
- **Slow inference** - Enable performance mode
- **Import errors** - Verify all dependencies installed

## Support

For detailed instructions, see `DEPLOYMENT_JETSON.md`.

For issues:
1. Run `test_deployment.py` to diagnose
2. Check error messages
3. Verify dependencies
4. Review deployment guide

## License

Please refer to the LICENSE file for licensing information.
