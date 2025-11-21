#!/usr/bin/env python3
#*----------------------------------------------------------------------------*
#* ONNX Runtime Inference Script for Jetson Xavier NX                        *
#* Faster inference using ONNX Runtime (optional)                            *
#*----------------------------------------------------------------------------*

"""
ONNX Runtime Inference Script for Jetson Xavier NX
This script uses ONNX Runtime for faster inference compared to TensorFlow/Keras.
Requires: pip3 install onnxruntime-gpu
"""

import numpy as np
import os
import argparse
import sys

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime not installed.")
    print("Install with: pip3 install onnxruntime-gpu")
    sys.exit(1)

from eeg_reduction import eeg_reduction


def load_onnx_model(model_path, providers=None):
    """
    Load ONNX model for inference
    
    Args:
        model_path: Path to the .onnx model file
        providers: List of execution providers (default: ['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    Returns:
        ONNX Runtime InferenceSession
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if providers is None:
        # Try CUDA first, fallback to CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    print(f"Loading ONNX model from: {model_path}")
    
    # Create inference session
    session = ort.InferenceSession(
        model_path,
        providers=providers
    )
    
    # Get input/output info
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    print(f"Model loaded successfully!")
    print(f"Input name: {input_name}")
    print(f"Input shape: {input_shape}")
    print(f"Output name: {output_name}")
    print(f"Providers: {session.get_providers()}")
    
    return session, input_name, output_name


def preprocess_eeg_data(X, n_ds=1, n_ch=64, T=3):
    """
    Preprocess EEG data for inference
    
    Args:
        X: Raw EEG data array (n_trials, n_channels, n_samples)
        n_ds: Downsampling factor
        n_ch: Number of channels
        T: Time window duration in seconds
    
    Returns:
        Preprocessed data ready for model input
    """
    # Apply EEG reduction
    X_processed = eeg_reduction(X, n_ds=n_ds, n_ch=n_ch, T=T)
    
    # Expand dimensions to match expected input: (batch, channels, samples, 1)
    X_processed = np.expand_dims(X_processed, axis=-1)
    
    # Convert to float32 for ONNX Runtime
    X_processed = X_processed.astype(np.float32)
    
    return X_processed


def predict_onnx(session, input_name, output_name, X, batch_size=16):
    """
    Run inference using ONNX Runtime
    
    Args:
        session: ONNX Runtime InferenceSession
        input_name: Name of input tensor
        output_name: Name of output tensor
        X: Preprocessed EEG data
        batch_size: Batch size for inference
    
    Returns:
        Predictions (probabilities) and predicted classes
    """
    n_samples = len(X)
    all_predictions = []
    
    # Process in batches
    for i in range(0, n_samples, batch_size):
        batch = X[i:i+batch_size]
        
        # Run inference
        outputs = session.run([output_name], {input_name: batch})
        predictions_batch = outputs[0]
        all_predictions.append(predictions_batch)
    
    # Concatenate all predictions
    predictions = np.concatenate(all_predictions, axis=0)
    
    # Get predicted classes
    predicted_classes = np.argmax(predictions, axis=1)
    
    return predictions, predicted_classes


def main():
    parser = argparse.ArgumentParser(
        description='ONNX Runtime Inference for Jetson Xavier NX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model inference
  python3 inference_onnx.py --model output.onnx --data sample_data.npy
  
  # Custom batch size
  python3 inference_onnx.py --model output.onnx --data sample_data.npy --batch_size 32
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to ONNX model file (.onnx)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input EEG data file (.npy)')
    parser.add_argument('--n_ds', type=int, default=1,
                        help='Downsampling factor (default: 1)')
    parser.add_argument('--n_ch', type=int, default=64,
                        help='Number of channels (default: 64)')
    parser.add_argument('--T', type=int, default=3,
                        help='Time window duration in seconds (default: 3)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference (default: 16)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file to save predictions')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU execution (disable GPU)')
    
    args = parser.parse_args()
    
    # Set providers
    if args.cpu:
        providers = ['CPUExecutionProvider']
    else:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    # Load model
    try:
        session, input_name, output_name = load_onnx_model(args.model, providers=providers)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load data
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)
    
    print(f"\nLoading data from: {args.data}")
    X = np.load(args.data)
    print(f"Data shape: {X.shape}")
    
    # Preprocess data
    print("Preprocessing data...")
    X_processed = preprocess_eeg_data(X, n_ds=args.n_ds, n_ch=args.n_ch, T=args.T)
    print(f"Preprocessed shape: {X_processed.shape}")
    
    # Run inference
    print(f"\nRunning inference (batch_size={args.batch_size})...")
    import time
    start_time = time.time()
    
    predictions, predicted_classes = predict_onnx(
        session, input_name, output_name,
        X_processed, batch_size=args.batch_size
    )
    
    inference_time = time.time() - start_time
    
    # Print results
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Inference time: {inference_time:.4f} seconds")
    print(f"Throughput: {len(X_processed)/inference_time:.2f} samples/second")
    
    for i in range(len(predicted_classes)):
        print(f"\nSample {i+1}:")
        print(f"  Predicted class: {predicted_classes[i]}")
        print(f"  Probabilities: {predictions[i]}")
        print(f"  Confidence: {predictions[i][predicted_classes[i]]:.4f}")
    
    # Save results if requested
    if args.output:
        results = {
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'inference_time': inference_time,
            'model_file': args.model
        }
        np.savez(args.output, **results)
        print(f"\nResults saved to: {args.output}")
    
    print("\nInference completed successfully!")


if __name__ == '__main__':
    main()
