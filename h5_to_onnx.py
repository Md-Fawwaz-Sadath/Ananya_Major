#!/usr/bin/env python3
"""
H5 to ONNX Conversion Script for EEGNet Model
This script converts Keras .h5 model files to ONNX format for deployment on Jetson Xavier NX
"""

import os
import sys
import numpy as np
import argparse
import tensorflow as tf
from keras.models import load_model
import keras.backend as K

def convert_h5_to_onnx(h5_model_path, onnx_output_path, opset_version=11):
    """
    Convert Keras H5 model to ONNX format
    
    Args:
        h5_model_path: Path to input .h5 model file
        onnx_output_path: Path to save output .onnx file
        opset_version: ONNX opset version (default: 11 for compatibility)
    """
    print(f"Loading Keras model from: {h5_model_path}")
    
    try:
        # Load the Keras model
        model = load_model(h5_model_path)
        print("Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        # Import tf2onnx for conversion
        try:
            import tf2onnx
        except ImportError:
            print("\nERROR: tf2onnx not installed!")
            print("Please install it with: pip3 install tf2onnx")
            sys.exit(1)
        
        # Get input/output specs
        input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name='input')]
        
        # Convert to ONNX
        print(f"\nConverting to ONNX format (opset {opset_version})...")
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=opset_version,
            output_path=onnx_output_path
        )
        
        print(f"\n✓ ONNX model saved successfully to: {onnx_output_path}")
        
        # Verify the ONNX model
        try:
            import onnx
            onnx_model_check = onnx.load(onnx_output_path)
            onnx.checker.check_model(onnx_model_check)
            print("✓ ONNX model verification passed!")
        except ImportError:
            print("Note: Install onnx package to verify the model: pip3 install onnx")
        except Exception as e:
            print(f"Warning: ONNX model verification failed: {e}")
        
        # Clear session
        K.clear_session()
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert Keras H5 model to ONNX format')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input .h5 model file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to output .onnx file (default: same as input with .onnx extension)')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset version (default: 11)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Set output path if not specified
    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + '.onnx'
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("H5 to ONNX Conversion")
    print("=" * 70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Opset:  {args.opset}")
    print("=" * 70)
    print()
    
    # Perform conversion
    success = convert_h5_to_onnx(args.input, args.output, args.opset)
    
    if success:
        print("\n" + "=" * 70)
        print("Conversion completed successfully!")
        print("=" * 70)
        print(f"\nNext steps:")
        print(f"1. Test ONNX model: python3 test_onnx.py --model {args.output}")
        print(f"2. Convert to TensorRT: python3 onnx_to_tensorrt.py --input {args.output}")
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("Conversion failed!")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
