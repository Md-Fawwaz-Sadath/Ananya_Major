#!/usr/bin/env python3
"""
Jetson Xavier NX Inference Script for EEGNet BCI
Real-time motor imagery classification using ONNX Runtime

Copyright (C) 2020 ETH Zurich, Switzerland
SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
import onnxruntime as ort
import argparse
import time
import sys
from eeg_reduction import eeg_reduction

class EEGNetInference:
    """
    EEGNet inference wrapper for Jetson Xavier NX
    Supports both ONNX and Keras (.h5) models
    """
    
    def __init__(self, model_path, model_type='onnx', use_cuda=True):
        """
        Initialize the inference engine
        
        Args:
            model_path: Path to model file (.onnx or .h5)
            model_type: 'onnx' or 'keras'
            use_cuda: Use CUDA acceleration if available
        """
        self.model_path = model_path
        self.model_type = model_type
        self.use_cuda = use_cuda
        
        if model_type == 'onnx':
            self._load_onnx_model()
        elif model_type == 'keras':
            self._load_keras_model()
        else:
            raise ValueError("model_type must be 'onnx' or 'keras'")
    
    def _load_onnx_model(self):
        """Load ONNX model with CUDA support"""
        providers = ['CPUExecutionProvider']
        
        if self.use_cuda:
            # Check if CUDA is available
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print("✓ CUDA acceleration enabled for ONNX Runtime")
            else:
                print("⚠ CUDA not available, using CPU")
        
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Get input shape
        input_shape = self.session.get_inputs()[0].shape
        print(f"Model loaded: {self.model_path}")
        print(f"Input shape: {input_shape}")
        print(f"Output shape: {self.session.get_outputs()[0].shape}")
    
    def _load_keras_model(self):
        """Load Keras .h5 model"""
        try:
            from keras.models import load_model
            from tensorflow.keras.constraints import max_norm
            import tensorflow as tf
            
            # Custom objects for loading
            custom_objects = {'max_norm': max_norm}
            
            self.model = load_model(self.model_path, custom_objects=custom_objects)
            print(f"✓ Keras model loaded: {self.model_path}")
            print(f"Input shape: {self.model.input_shape}")
            print(f"Output shape: {self.model.output_shape}")
        except Exception as e:
            print(f"✗ Error loading Keras model: {e}")
            sys.exit(1)
    
    def preprocess(self, eeg_data, n_ds=1, n_ch=64, T=3, fs=160):
        """
        Preprocess EEG data before inference
        
        Args:
            eeg_data: Raw EEG data, shape (n_trials, 64, n_samples)
            n_ds: Downsampling factor
            n_ch: Number of channels
            T: Time window in seconds
            fs: Sampling frequency in Hz
        
        Returns:
            Preprocessed data ready for model input
        """
        # Apply EEG reduction (channel selection, downsampling, time window)
        X = eeg_reduction(eeg_data, n_ds=n_ds, n_ch=n_ch, T=T, fs=fs)
        
        # Add channel dimension for model input
        X = np.expand_dims(X, axis=-1)
        
        return X.astype(np.float32)
    
    def predict(self, X):
        """
        Run inference on preprocessed data
        
        Args:
            X: Preprocessed EEG data, shape (batch_size, channels, samples, 1)
        
        Returns:
            predictions: Class probabilities, shape (batch_size, num_classes)
        """
        if self.model_type == 'onnx':
            predictions = self.session.run(
                [self.output_name],
                {self.input_name: X}
            )[0]
        else:
            predictions = self.model.predict(X, verbose=0)
        
        return predictions
    
    def predict_with_timing(self, X):
        """
        Run inference and return predictions with timing info
        
        Returns:
            predictions, inference_time_ms
        """
        start_time = time.time()
        predictions = self.predict(X)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return predictions, inference_time
    
    def classify(self, eeg_data, n_ds=1, n_ch=64, T=3, fs=160, class_names=None):
        """
        Complete pipeline: preprocess + inference + decode
        
        Args:
            eeg_data: Raw EEG data
            class_names: Optional list of class names for output
        
        Returns:
            results: List of dicts with predictions for each trial
        """
        # Preprocess
        X = self.preprocess(eeg_data, n_ds=n_ds, n_ch=n_ch, T=T, fs=fs)
        
        # Inference with timing
        predictions, inference_time = self.predict_with_timing(X)
        
        # Decode results
        results = []
        for i, pred in enumerate(predictions):
            predicted_class = np.argmax(pred)
            confidence = pred[predicted_class]
            
            result = {
                'trial': i,
                'predicted_class': int(predicted_class),
                'confidence': float(confidence),
                'probabilities': pred.tolist(),
                'inference_time_ms': inference_time / len(predictions)
            }
            
            if class_names:
                result['class_name'] = class_names[predicted_class]
            
            results.append(result)
        
        return results


def demo_inference(model_path, model_type='onnx', num_samples=10):
    """
    Demonstration of inference with synthetic data
    """
    print("="*60)
    print("EEGNet Jetson Xavier NX Inference Demo")
    print("="*60)
    
    # Initialize inference engine
    engine = EEGNetInference(model_path, model_type=model_type, use_cuda=True)
    
    # Generate synthetic EEG data for testing
    # Shape: (num_samples, 64_channels, 480_time_points)
    # 480 time points = 3 seconds at 160 Hz
    print(f"\nGenerating {num_samples} synthetic EEG trials...")
    synthetic_data = np.random.randn(num_samples, 64, 480).astype(np.float32)
    
    # Class names for 4-class motor imagery
    class_names = ['Left Hand', 'Right Hand', 'Both Feet', 'Rest']
    
    # Run classification
    print("\nRunning inference...")
    results = engine.classify(
        synthetic_data,
        n_ds=1,
        n_ch=64,
        T=3,
        fs=160,
        class_names=class_names
    )
    
    # Display results
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    
    total_time = 0
    for result in results:
        print(f"\nTrial {result['trial'] + 1}:")
        print(f"  Predicted: {result.get('class_name', result['predicted_class'])}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Inference time: {result['inference_time_ms']:.2f} ms")
        total_time += result['inference_time_ms']
    
    avg_time = total_time / len(results)
    fps = 1000 / avg_time
    
    print("\n" + "="*60)
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    print("="*60)


def benchmark_model(model_path, model_type='onnx', num_iterations=100):
    """
    Benchmark model performance on Jetson
    """
    print("="*60)
    print("EEGNet Jetson Xavier NX Benchmark")
    print("="*60)
    
    engine = EEGNetInference(model_path, model_type=model_type, use_cuda=True)
    
    # Prepare test data
    test_data = np.random.randn(1, 64, 480).astype(np.float32)
    X = engine.preprocess(test_data)
    
    # Warmup
    print("\nWarming up (10 iterations)...")
    for _ in range(10):
        _ = engine.predict(X)
    
    # Benchmark
    print(f"\nRunning benchmark ({num_iterations} iterations)...")
    times = []
    
    for i in range(num_iterations):
        start = time.time()
        _ = engine.predict(X)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{num_iterations}")
    
    # Statistics
    times = np.array(times)
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Mean inference time: {np.mean(times):.2f} ms")
    print(f"Std deviation: {np.std(times):.2f} ms")
    print(f"Min: {np.min(times):.2f} ms")
    print(f"Max: {np.max(times):.2f} ms")
    print(f"Median: {np.median(times):.2f} ms")
    print(f"95th percentile: {np.percentile(times, 95):.2f} ms")
    print(f"Throughput: {1000/np.mean(times):.2f} inferences/sec")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='EEGNet inference on Jetson Xavier NX'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model file (.onnx or .h5)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='onnx',
        choices=['onnx', 'keras'],
        help='Model type (default: onnx)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='demo',
        choices=['demo', 'benchmark'],
        help='Run mode: demo or benchmark (default: demo)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples for demo (default: 10)'
    )
    parser.add_argument(
        '--num-iterations',
        type=int,
        default=100,
        help='Number of iterations for benchmark (default: 100)'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input .npy file with EEG data'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    import os
    if not os.path.exists(args.model):
        print(f"✗ Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Run selected mode
    if args.mode == 'demo':
        demo_inference(args.model, args.model_type, args.num_samples)
    elif args.mode == 'benchmark':
        benchmark_model(args.model, args.model_type, args.num_iterations)
