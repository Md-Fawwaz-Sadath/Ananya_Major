#!/usr/bin/env python3
"""
Local test script for EEGNet model before deploying to Jetson
Tests model loading and basic inference functionality
"""

import numpy as np
import sys
import os

def test_onnx_model():
    """Test ONNX model loading and inference"""
    print("\n" + "="*60)
    print("Testing ONNX Model")
    print("="*60)
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("✗ ONNX Runtime not installed")
        print("  Install: pip install onnxruntime")
        return False
    
    model_path = "output.onnx"
    if not os.path.exists(model_path):
        print(f"✗ Model file not found: {model_path}")
        return False
    
    try:
        # Load model
        session = ort.InferenceSession(model_path)
        print(f"✓ Model loaded: {model_path}")
        
        # Get input/output info
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape
        
        print(f"  Input name: {input_name}")
        print(f"  Input shape: {input_shape}")
        print(f"  Output shape: {output_shape}")
        
        # Create test data
        # Assuming input shape is [batch, channels, samples, 1]
        test_input = np.random.randn(1, 64, 480, 1).astype(np.float32)
        
        # Run inference
        output = session.run(None, {input_name: test_input})[0]
        
        print(f"✓ Inference successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Output (probabilities): {output[0]}")
        print(f"  Predicted class: {np.argmax(output[0])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_keras_model():
    """Test Keras model loading"""
    print("\n" + "="*60)
    print("Testing Keras Model")
    print("="*60)
    
    try:
        from keras.models import load_model
        from tensorflow.keras.constraints import max_norm
    except ImportError:
        print("✗ Keras/TensorFlow not installed")
        print("  Install: pip install tensorflow keras")
        return False
    
    # Look for .h5 models
    h5_files = [f for f in os.listdir('.') if f.endswith('.h5')]
    
    if not h5_files:
        print("⚠ No .h5 model files found in current directory")
        return False
    
    model_path = h5_files[0]
    print(f"Found model: {model_path}")
    
    try:
        # Load model
        custom_objects = {'max_norm': max_norm}
        model = load_model(model_path, custom_objects=custom_objects)
        print(f"✓ Model loaded: {model_path}")
        
        # Model info
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        
        # Create test data
        test_input = np.random.randn(1, 64, 480, 1).astype(np.float32)
        
        # Run inference
        output = model.predict(test_input, verbose=0)
        
        print(f"✓ Inference successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Output (probabilities): {output[0]}")
        print(f"  Predicted class: {np.argmax(output[0])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_preprocessing():
    """Test EEG preprocessing function"""
    print("\n" + "="*60)
    print("Testing EEG Preprocessing")
    print("="*60)
    
    try:
        from eeg_reduction import eeg_reduction
    except ImportError:
        print("✗ eeg_reduction.py not found")
        return False
    
    try:
        # Create synthetic EEG data
        # Shape: (n_trials, 64_channels, 480_samples)
        test_data = np.random.randn(5, 64, 480).astype(np.float32)
        print(f"Input shape: {test_data.shape}")
        
        # Apply reduction
        processed = eeg_reduction(test_data, n_ds=1, n_ch=64, T=3, fs=160)
        print(f"✓ Preprocessing successful")
        print(f"  Output shape: {processed.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("  EEGNet BCI - Local Test Suite")
    print("="*60)
    
    results = {
        'ONNX Model': test_onnx_model(),
        'Keras Model': test_keras_model(),
        'Preprocessing': test_preprocessing()
    }
    
    print("\n" + "="*60)
    print("  Test Results Summary")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20s} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("  ✓ All tests passed! Ready for Jetson deployment.")
    else:
        print("  ⚠ Some tests failed. Check errors above.")
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
