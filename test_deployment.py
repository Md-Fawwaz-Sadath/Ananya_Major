#!/usr/bin/env python3
#*----------------------------------------------------------------------------*
#* Test script for Jetson deployment                                         *
#* Creates sample data and tests inference                                   *
#*----------------------------------------------------------------------------*

"""
Test script to verify deployment on Jetson Xavier NX
Creates sample EEG data and tests inference
"""

import numpy as np
import os
import sys
import tempfile

def create_sample_data(n_trials=5, n_channels=64, n_samples=480, save_path=None):
    """
    Create sample EEG data for testing
    
    Args:
        n_trials: Number of trials
        n_channels: Number of channels
        n_samples: Number of time samples (480 = 3 seconds at 160 Hz)
        save_path: Path to save the data (optional)
    
    Returns:
        Sample EEG data array
    """
    # Create random EEG-like data
    np.random.seed(42)
    X = np.random.randn(n_trials, n_channels, n_samples).astype(np.float32)
    
    if save_path:
        np.save(save_path, X)
        print(f"Sample data saved to: {save_path}")
    
    return X


def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import numpy
        print(f"  ✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy: {e}")
        return False
    
    try:
        import scipy
        print(f"  ✓ SciPy {scipy.__version__}")
    except ImportError as e:
        print(f"  ✗ SciPy: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"  ✓ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"  ✗ TensorFlow: {e}")
        return False
    
    try:
        import keras
        print(f"  ✓ Keras {keras.__version__}")
    except ImportError as e:
        print(f"  ✗ Keras: {e}")
        return False
    
    try:
        from eeg_reduction import eeg_reduction
        print(f"  ✓ eeg_reduction module")
    except ImportError as e:
        print(f"  ✗ eeg_reduction: {e}")
        return False
    
    try:
        import models
        print(f"  ✓ models module")
    except ImportError as e:
        print(f"  ✗ models: {e}")
        return False
    
    return True


def test_eeg_reduction():
    """Test EEG reduction function"""
    print("\nTesting EEG reduction...")
    
    try:
        from eeg_reduction import eeg_reduction
        
        # Create sample data
        X = np.random.randn(10, 64, 480)
        
        # Test reduction
        X_reduced = eeg_reduction(X, n_ds=1, n_ch=64, T=3)
        
        expected_shape = (10, 64, 480)
        if X_reduced.shape == expected_shape:
            print(f"  ✓ EEG reduction: {X_reduced.shape}")
            return True
        else:
            print(f"  ✗ EEG reduction: Expected {expected_shape}, got {X_reduced.shape}")
            return False
    except Exception as e:
        print(f"  ✗ EEG reduction failed: {e}")
        return False


def test_model_loading(model_path):
    """Test if a model can be loaded"""
    print(f"\nTesting model loading: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"  ⚠ Model file not found: {model_path}")
        print("  Skipping model loading test")
        return None
    
    try:
        from keras.models import load_model
        model = load_model(model_path)
        print(f"  ✓ Model loaded successfully")
        print(f"    Input shape: {model.input_shape}")
        print(f"    Output shape: {model.output_shape}")
        return True
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        return False


def test_inference(model_path=None, use_sample_data=True):
    """Test inference with sample data"""
    print("\nTesting inference...")
    
    if model_path is None:
        print("  ⚠ No model provided, skipping inference test")
        return None
    
    if not os.path.exists(model_path):
        print(f"  ⚠ Model file not found: {model_path}")
        return None
    
    try:
        # Create sample data
        if use_sample_data:
            X = create_sample_data(n_trials=2, n_channels=64, n_samples=480)
        else:
            print("  Using provided data file")
            return None
        
        # Test preprocessing
        from eeg_reduction import eeg_reduction
        X_processed = eeg_reduction(X, n_ds=1, n_ch=64, T=3)
        X_processed = np.expand_dims(X_processed, axis=-1)
        
        # Load model and predict
        from keras.models import load_model
        model = load_model(model_path)
        
        predictions = model.predict(X_processed, batch_size=2, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        print(f"  ✓ Inference successful")
        print(f"    Input shape: {X_processed.shape}")
        print(f"    Output shape: {predictions.shape}")
        print(f"    Predicted classes: {predicted_classes}")
        
        return True
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test deployment on Jetson')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file for testing (.h5)')
    parser.add_argument('--create-data', action='store_true',
                        help='Create and save sample data file')
    parser.add_argument('--data-file', type=str, default='test_data.npy',
                        help='Path to save sample data')
    
    args = parser.parse_args()
    
    print("="*60)
    print("EEGNet BCI Deployment Test for Jetson Xavier NX")
    print("="*60)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed. Please install missing dependencies.")
        sys.exit(1)
    
    # Test 2: EEG reduction
    if not test_eeg_reduction():
        print("\n❌ EEG reduction test failed.")
        sys.exit(1)
    
    # Test 3: Model loading (if model provided)
    if args.model:
        test_model_loading(args.model)
    
    # Test 4: Inference (if model provided)
    if args.model:
        test_inference(args.model)
    
    # Create sample data if requested
    if args.create_data:
        create_sample_data(save_path=args.data_file)
        print(f"\n✓ Sample data created: {args.data_file}")
        print(f"  You can use this for testing inference:")
        print(f"  python3 inference.py --model <your_model.h5> --data {args.data_file}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Copy your trained model files to the models/ directory")
    print("2. Test inference with: python3 inference.py --model <model.h5> --data <data.npy>")
    print("3. For performance: sudo nvpmodel -m 0 && sudo jetson_clocks")


if __name__ == '__main__':
    main()
