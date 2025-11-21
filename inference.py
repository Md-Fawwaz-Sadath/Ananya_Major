#!/usr/bin/env python3
#*----------------------------------------------------------------------------*
#* Copyright (C) 2020 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Inference script optimized for Jetson Xavier NX                            *
#*----------------------------------------------------------------------------*

"""
EEGNet Inference Script for Jetson Xavier NX
This script performs real-time or batch inference using pre-trained EEGNet models.
"""

import numpy as np
import os
import argparse
import time
from keras.models import load_model
from keras import backend as K
import tensorflow as tf

# Import project modules
import models as models
from eeg_reduction import eeg_reduction

# Configure GPU for Jetson
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configure TensorFlow for Jetson
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # Allow GPU memory growth
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # Limit GPU memory usage
sess = tf.Session(config=config)
K.set_session(sess)


class EEGNetInference:
    """EEGNet inference class for real-time and batch processing"""
    
    def __init__(self, model_path, num_classes=4, n_ds=1, n_ch=64, T=3):
        """
        Initialize EEGNet inference engine
        
        Args:
            model_path: Path to trained .h5 model file
            num_classes: Number of classes (default: 4)
            n_ds: Downsampling factor (default: 1)
            n_ch: Number of channels (default: 64)
            T: Time window in seconds (default: 3)
        """
        self.num_classes = num_classes
        self.n_ds = n_ds
        self.n_ch = n_ch
        self.T = T
        self.fs = 160  # Sampling frequency
        
        # Calculate model parameters
        self.kernLength = int(np.ceil(128/n_ds))
        self.poolLength = int(np.ceil(8/n_ds))
        self.n_samples = int(np.ceil(T * self.fs / n_ds))
        
        print(f"Loading model from: {model_path}")
        self.model = load_model(model_path)
        print("Model loaded successfully!")
        
        # Warm up the model with dummy data
        self._warmup()
    
    def _warmup(self):
        """Warm up the model with dummy data for faster first inference"""
        dummy_input = np.random.randn(1, self.n_ch, self.n_samples, 1)
        _ = self.model.predict(dummy_input, batch_size=1, verbose=0)
        print("Model warmed up!")
    
    def preprocess(self, X):
        """
        Preprocess EEG data for inference
        
        Args:
            X: Raw EEG data array of shape (n_trials, n_channels, n_samples)
               or (n_channels, n_samples) for single trial
               
        Returns:
            Preprocessed data ready for model input
        """
        # Handle single trial case
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=0)
        
        # Apply EEG reduction (downsampling, channel selection, time window)
        X_processed = eeg_reduction(X, n_ds=self.n_ds, n_ch=self.n_ch, T=self.T, fs=self.fs)
        
        # Expand dimensions for model input (add channel dimension)
        X_processed = np.expand_dims(X_processed, axis=-1)
        
        return X_processed
    
    def predict(self, X, batch_size=16, verbose=0):
        """
        Run inference on preprocessed data
        
        Args:
            X: Preprocessed EEG data
            batch_size: Batch size for inference
            verbose: Verbosity level
            
        Returns:
            Predictions array of shape (n_trials, num_classes)
        """
        predictions = self.model.predict(X, batch_size=batch_size, verbose=verbose)
        return predictions
    
    def predict_proba(self, X, batch_size=16):
        """Get class probabilities"""
        return self.predict(X, batch_size=batch_size)
    
    def predict_class(self, X, batch_size=16):
        """Get predicted class indices"""
        proba = self.predict(X, batch_size=batch_size)
        return np.argmax(proba, axis=1)
    
    def predict_single(self, X):
        """
        Predict on a single trial with timing information
        
        Args:
            X: Single trial EEG data (n_channels, n_samples) or raw data
            
        Returns:
            dict with predictions, probabilities, and timing info
        """
        start_time = time.time()
        
        # Preprocess
        X_processed = self.preprocess(X)
        
        # Predict
        proba = self.predict(X_processed, batch_size=1, verbose=0)
        pred_class = np.argmax(proba[0])
        confidence = np.max(proba[0])
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            'class': int(pred_class),
            'confidence': float(confidence),
            'probabilities': proba[0].tolist(),
            'inference_time_ms': inference_time
        }
    
    def cleanup(self):
        """Clean up resources"""
        K.clear_session()


def load_numpy_data(filepath):
    """Load EEG data from numpy file"""
    data = np.load(filepath)
    # Handle different numpy file formats
    if isinstance(data, np.lib.npyio.NpzFile):
        # .npz file - try common keys
        keys = list(data.keys())
        if 'X' in keys:
            return data['X']
        elif 'X_Train' in keys:
            return data['X_Train']
        elif 'data' in keys:
            return data['data']
        else:
            return data[keys[0]]  # Return first array
    else:
        return data


def main():
    parser = argparse.ArgumentParser(description='EEGNet Inference on Jetson Xavier NX')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.h5 file)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input data (.npy or .npz file)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save predictions (optional)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for inference (default: 16)')
    parser.add_argument('--num-classes', type=int, default=4,
                        help='Number of classes (default: 4)')
    parser.add_argument('--n-ds', type=int, default=1,
                        help='Downsampling factor (default: 1)')
    parser.add_argument('--n-ch', type=int, default=64,
                        help='Number of channels (default: 64)')
    parser.add_argument('--T', type=float, default=3.0,
                        help='Time window in seconds (default: 3.0)')
    parser.add_argument('--realtime', action='store_true',
                        help='Run in real-time mode (single trial inference)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark test')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    print("=" * 60)
    print("EEGNet Inference Engine for Jetson Xavier NX")
    print("=" * 60)
    
    try:
        inference_engine = EEGNetInference(
            model_path=args.model,
            num_classes=args.num_classes,
            n_ds=args.n_ds,
            n_ch=args.n_ch,
            T=args.T
        )
        
        # Load data
        print(f"\nLoading data from: {args.data}")
        X = load_numpy_data(args.data)
        print(f"Data shape: {X.shape}")
        
        if args.realtime:
            # Real-time mode: process single trials
            print("\nRunning in real-time mode...")
            if len(X.shape) == 3:
                # Process each trial separately
                results = []
                for i in range(X.shape[0]):
                    result = inference_engine.predict_single(X[i])
                    results.append(result)
                    print(f"Trial {i+1}: Class={result['class']}, "
                          f"Confidence={result['confidence']:.4f}, "
                          f"Time={result['inference_time_ms']:.2f}ms")
                
                if args.output:
                    np.save(args.output, results)
                    print(f"\nResults saved to: {args.output}")
            else:
                result = inference_engine.predict_single(X)
                print(f"Prediction: Class={result['class']}, "
                      f"Confidence={result['confidence']:.4f}, "
                      f"Time={result['inference_time_ms']:.2f}ms")
        
        elif args.benchmark:
            # Benchmark mode
            print("\nRunning benchmark...")
            X_processed = inference_engine.preprocess(X)
            
            # Warm up
            _ = inference_engine.predict(X_processed[:min(10, len(X_processed))], 
                                        batch_size=args.batch_size, verbose=0)
            
            # Benchmark
            n_iterations = 100
            start_time = time.time()
            for _ in range(n_iterations):
                _ = inference_engine.predict(X_processed[:min(10, len(X_processed))], 
                                            batch_size=args.batch_size, verbose=0)
            total_time = time.time() - start_time
            
            avg_time = (total_time / n_iterations) * 1000  # ms
            throughput = (10 * n_iterations) / total_time  # samples/sec
            
            print(f"\nBenchmark Results:")
            print(f"  Average inference time: {avg_time:.2f} ms")
            print(f"  Throughput: {throughput:.2f} samples/sec")
        
        else:
            # Batch mode
            print("\nRunning batch inference...")
            X_processed = inference_engine.preprocess(X)
            print(f"Preprocessed data shape: {X_processed.shape}")
            
            start_time = time.time()
            predictions = inference_engine.predict(X_processed, 
                                                  batch_size=args.batch_size, 
                                                  verbose=1)
            inference_time = time.time() - start_time
            
            predicted_classes = np.argmax(predictions, axis=1)
            confidences = np.max(predictions, axis=1)
            
            print(f"\nInference completed in {inference_time:.2f} seconds")
            print(f"Processed {len(X_processed)} samples")
            print(f"Average time per sample: {(inference_time/len(X_processed))*1000:.2f} ms")
            print(f"\nPredictions:")
            for i, (cls, conf) in enumerate(zip(predicted_classes, confidences)):
                print(f"  Sample {i+1}: Class={cls}, Confidence={conf:.4f}")
            
            if args.output:
                results = {
                    'predictions': predicted_classes,
                    'probabilities': predictions,
                    'confidences': confidences
                }
                np.savez(args.output, **results)
                print(f"\nResults saved to: {args.output}")
        
        # Cleanup
        inference_engine.cleanup()
        print("\nDone!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
