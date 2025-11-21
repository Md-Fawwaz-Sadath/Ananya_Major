#!/usr/bin/env python3
#*----------------------------------------------------------------------------*
#* Test script for Jetson deployment                                           *
#*----------------------------------------------------------------------------*

"""
Test script to verify deployment on Jetson Xavier NX
This script checks all dependencies and tests basic functionality
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("=" * 60)
    print("Testing Imports")
    print("=" * 60)
    
    modules = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('sklearn', 'scikit-learn'),
        ('h5py', 'h5py'),
        ('keras', 'Keras'),
        ('tensorflow', 'TensorFlow'),
    ]
    
    failed = []
    for module_name, display_name in modules:
        try:
            mod = __import__(module_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {display_name}: {version}")
        except ImportError as e:
            print(f"✗ {display_name}: FAILED - {e}")
            failed.append(display_name)
    
    if failed:
        print(f"\nFailed imports: {', '.join(failed)}")
        return False
    else:
        print("\nAll imports successful!")
        return True


def test_tensorflow_gpu():
    """Test TensorFlow GPU availability"""
    print("\n" + "=" * 60)
    print("Testing TensorFlow GPU")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Check GPU availability
        gpu_available = tf.test.is_gpu_available()
        if gpu_available:
            print("✓ GPU is available")
            
            # List GPUs
            gpus = tf.config.list_physical_devices('GPU')
            print(f"  Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"    GPU {i}: {gpu.name}")
        else:
            print("✗ GPU is NOT available")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error testing TensorFlow GPU: {e}")
        return False


def test_project_modules():
    """Test if project modules can be imported"""
    print("\n" + "=" * 60)
    print("Testing Project Modules")
    print("=" * 60)
    
    modules = ['models', 'eeg_reduction', 'inference']
    failed = []
    
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}.py")
        except ImportError as e:
            print(f"✗ {module_name}.py: FAILED - {e}")
            failed.append(module_name)
    
    if failed:
        print(f"\nFailed modules: {', '.join(failed)}")
        return False
    else:
        print("\nAll project modules loaded successfully!")
        return True


def test_model_loading():
    """Test if a model can be loaded"""
    print("\n" + "=" * 60)
    print("Testing Model Loading")
    print("=" * 60)
    
    import glob
    model_files = glob.glob('models/*.h5')
    
    if not model_files:
        print("⚠ No model files found in models/ directory")
        print("  Skipping model loading test")
        return True
    
    try:
        from keras.models import load_model
        model_path = model_files[0]
        print(f"Loading model: {model_path}")
        model = load_model(model_path)
        print(f"✓ Model loaded successfully!")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False


def test_inference():
    """Test inference with dummy data"""
    print("\n" + "=" * 60)
    print("Testing Inference")
    print("=" * 60)
    
    import glob
    import numpy as np
    model_files = glob.glob('models/*.h5')
    
    if not model_files:
        print("⚠ No model files found, skipping inference test")
        return True
    
    try:
        from inference import EEGNetInference
        
        model_path = model_files[0]
        print(f"Initializing inference engine with: {model_path}")
        engine = EEGNetInference(model_path, num_classes=4, n_ch=64, T=3.0)
        
        # Generate dummy data
        print("Generating dummy EEG data...")
        dummy_data = np.random.randn(64, 480)  # 64 channels, 480 samples (3 sec at 160 Hz)
        
        # Run inference
        print("Running inference...")
        result = engine.predict_single(dummy_data)
        
        print(f"✓ Inference successful!")
        print(f"  Predicted class: {result['class']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Inference time: {result['inference_time_ms']:.2f} ms")
        
        engine.cleanup()
        return True
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_system_info():
    """Display system information"""
    print("\n" + "=" * 60)
    print("System Information")
    print("=" * 60)
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # Check if running on Jetson
    if os.path.exists('/etc/nv_tegra_release'):
        with open('/etc/nv_tegra_release', 'r') as f:
            tegra_release = f.read().strip()
        print(f"Jetson detected: {tegra_release}")
    else:
        print("⚠ Not running on Jetson (or nv_tegra_release not found)")
    
    # Memory info
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"Total memory: {mem.total / (1024**3):.2f} GB")
        print(f"Available memory: {mem.available / (1024**3):.2f} GB")
    except ImportError:
        print("(Install psutil for memory info)")
    
    # CPU info
    try:
        import multiprocessing
        print(f"CPU cores: {multiprocessing.cpu_count()}")
    except:
        pass


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("EEGNet BCI Deployment Test for Jetson Xavier NX")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("TensorFlow GPU", test_tensorflow_gpu()))
    results.append(("Project Modules", test_project_modules()))
    results.append(("Model Loading", test_model_loading()))
    results.append(("Inference", test_inference()))
    
    # System info
    test_system_info()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Deployment is successful.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
