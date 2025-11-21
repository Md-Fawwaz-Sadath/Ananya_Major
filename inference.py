#!/usr/bin/env python3
#*----------------------------------------------------------------------------*
#* Copyright (C) 2020 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Inference script for Jetson Xavier NX deployment                           *
#*----------------------------------------------------------------------------*

"""
EEGNet Inference Script for Jetson Xavier NX
This script loads trained EEGNet models and performs inference on EEG data.
"""

import numpy as np
import os
import argparse
import sys
from keras.models import load_model
from keras import backend as K
from eeg_reduction import eeg_reduction

# Set GPU memory growth to avoid allocation issues on Jetson
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configure TensorFlow to use GPU memory growth
try:
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
except Exception as e:
    print(f"Warning: Could not configure GPU: {e}")


def load_eegnet_model(model_path, verbose=True):
    """
    Load a trained EEGNet model from .h5 file
    
    Args:
        model_path: Path to the .h5 model file
        verbose: Print model summary if True
    
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if verbose:
        print(f"Loading model from: {model_path}")
    
    model = load_model(model_path)
    
    if verbose:
        print(f"Model loaded successfully!")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
    
    return model


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
    # Apply EEG reduction (downsampling, channel selection, time window)
    X_processed = eeg_reduction(X, n_ds=n_ds, n_ch=n_ch, T=T)
    
    # Expand dimensions to match expected EEGNet input: (batch, channels, samples, 1)
    X_processed = np.expand_dims(X_processed, axis=-1)
    
    return X_processed


def predict(model, X, batch_size=16, verbose=1):
    """
    Run inference on preprocessed EEG data
    
    Args:
        model: Loaded Keras model
        X: Preprocessed EEG data
        batch_size: Batch size for inference
        verbose: Verbosity level
    
    Returns:
        Predictions (probabilities) and predicted classes
    """
    # Get predictions (probabilities)
    predictions = model.predict(X, batch_size=batch_size, verbose=verbose)
    
    # Get predicted classes
    predicted_classes = np.argmax(predictions, axis=1)
    
    return predictions, predicted_classes


def main():
    parser = argparse.ArgumentParser(
        description='EEGNet Inference for Jetson Xavier NX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model inference
  python3 inference.py --model models/global_class_4_ds1_nch64_T3_split_0.h5 --data sample_data.npy
  
  # Batch inference with multiple models (ensemble)
  python3 inference.py --model models/global_class_4_ds1_nch64_T3_split_*.h5 --data sample_data.npy --ensemble
  
  # Real-time inference mode
  python3 inference.py --model models/global_class_4_ds1_nch64_T3_split_0.h5 --realtime
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model file (.h5) or pattern for multiple models')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to input EEG data file (.npy)')
    parser.add_argument('--n_ds', type=int, default=1,
                        help='Downsampling factor (default: 1)')
    parser.add_argument('--n_ch', type=int, default=64,
                        help='Number of channels (default: 64)')
    parser.add_argument('--T', type=int, default=3,
                        help='Time window duration in seconds (default: 3)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference (default: 16)')
    parser.add_argument('--ensemble', action='store_true',
                        help='Use ensemble prediction from multiple models')
    parser.add_argument('--realtime', action='store_true',
                        help='Real-time inference mode (requires data streaming)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file to save predictions')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level (0, 1, or 2)')
    
    args = parser.parse_args()
    
    # Load model(s)
    import glob
    model_files = glob.glob(args.model) if '*' in args.model else [args.model]
    
    if len(model_files) == 0:
        print(f"Error: No model files found matching: {args.model}")
        sys.exit(1)
    
    models = []
    for model_file in model_files:
        try:
            model = load_eegnet_model(model_file, verbose=(args.verbose > 0))
            models.append(model)
        except Exception as e:
            print(f"Error loading model {model_file}: {e}")
            sys.exit(1)
    
    print(f"Loaded {len(models)} model(s)")
    
    # Load or prepare input data
    if args.realtime:
        print("Real-time mode: Waiting for EEG data stream...")
        print("Note: Real-time data acquisition needs to be implemented separately")
        # TODO: Implement real-time data streaming
        sys.exit(0)
    
    if args.data is None:
        print("Error: --data argument required (or use --realtime)")
        sys.exit(1)
    
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)
    
    # Load data
    print(f"Loading data from: {args.data}")
    X = np.load(args.data)
    print(f"Data shape: {X.shape}")
    
    # Preprocess data
    print("Preprocessing data...")
    X_processed = preprocess_eeg_data(X, n_ds=args.n_ds, n_ch=args.n_ch, T=args.T)
    print(f"Preprocessed shape: {X_processed.shape}")
    
    # Run inference
    print("Running inference...")
    if args.ensemble and len(models) > 1:
        # Ensemble prediction: average predictions from all models
        all_predictions = []
        for i, model in enumerate(models):
            pred, _ = predict(model, X_processed, batch_size=args.batch_size, verbose=args.verbose)
            all_predictions.append(pred)
        
        # Average predictions
        predictions = np.mean(all_predictions, axis=0)
        predicted_classes = np.argmax(predictions, axis=1)
        print(f"Ensemble prediction using {len(models)} models")
    else:
        # Single model prediction
        predictions, predicted_classes = predict(
            models[0], X_processed, 
            batch_size=args.batch_size, 
            verbose=args.verbose
        )
    
    # Print results
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    for i in range(len(predicted_classes)):
        print(f"Sample {i+1}:")
        print(f"  Predicted class: {predicted_classes[i]}")
        print(f"  Probabilities: {predictions[i]}")
        print(f"  Confidence: {predictions[i][predicted_classes[i]]:.4f}")
    
    print(f"\nOverall accuracy (if labels available): N/A")
    
    # Save results if requested
    if args.output:
        results = {
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'model_files': model_files
        }
        np.savez(args.output, **results)
        print(f"\nResults saved to: {args.output}")
    
    # Clean up
    K.clear_session()
    for model in models:
        del model
    
    print("\nInference completed successfully!")


if __name__ == '__main__':
    main()
