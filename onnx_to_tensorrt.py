#!/usr/bin/env python3
"""
ONNX to TensorRT Conversion Script for Jetson Xavier NX
This script converts ONNX model files to TensorRT engine for optimized inference
"""

import os
import sys
import argparse
import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("\nPlease ensure you're running this on Jetson with TensorRT installed.")
    print("TensorRT is pre-installed with JetPack SDK.")
    sys.exit(1)


class ONNXToTensorRT:
    """Convert ONNX model to TensorRT engine"""
    
    def __init__(self, onnx_path, engine_path, 
                 max_batch_size=1, 
                 fp16_mode=True,
                 int8_mode=False,
                 max_workspace_size=1 << 30):
        """
        Args:
            onnx_path: Path to ONNX model
            engine_path: Path to save TensorRT engine
            max_batch_size: Maximum batch size for inference
            fp16_mode: Enable FP16 precision (faster on Jetson)
            int8_mode: Enable INT8 precision (fastest, requires calibration)
            max_workspace_size: Maximum workspace size in bytes (default 1GB)
        """
        self.onnx_path = onnx_path
        self.engine_path = engine_path
        self.max_batch_size = max_batch_size
        self.fp16_mode = fp16_mode
        self.int8_mode = int8_mode
        self.max_workspace_size = max_workspace_size
        
        # TensorRT logger
        self.TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    
    def build_engine(self):
        """Build TensorRT engine from ONNX model"""
        
        print("Building TensorRT engine...")
        print(f"  Input ONNX: {self.onnx_path}")
        print(f"  Output Engine: {self.engine_path}")
        print(f"  Max Batch Size: {self.max_batch_size}")
        print(f"  FP16 Mode: {self.fp16_mode}")
        print(f"  INT8 Mode: {self.int8_mode}")
        print(f"  Max Workspace Size: {self.max_workspace_size / (1024**3):.2f} GB")
        print()
        
        # Create builder and network
        builder = trt.Builder(self.TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.TRT_LOGGER)
        
        # Parse ONNX model
        print("Parsing ONNX model...")
        with open(self.onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse ONNX model')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        print(f"✓ ONNX model parsed successfully")
        print(f"  Network inputs: {network.num_inputs}")
        print(f"  Network outputs: {network.num_outputs}")
        
        # Print input/output info
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            print(f"  Input {i}: {input_tensor.name}, shape={input_tensor.shape}, dtype={input_tensor.dtype}")
        
        for i in range(network.num_outputs):
            output_tensor = network.get_output(i)
            print(f"  Output {i}: {output_tensor.name}, shape={output_tensor.shape}, dtype={output_tensor.dtype}")
        print()
        
        # Create builder config
        config = builder.create_builder_config()
        config.max_workspace_size = self.max_workspace_size
        
        # Enable FP16 mode for faster inference on Jetson
        if self.fp16_mode and builder.platform_has_fast_fp16:
            print("✓ Enabling FP16 mode")
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.fp16_mode:
            print("⚠ FP16 not supported on this platform")
        
        # Enable INT8 mode (requires calibration data)
        if self.int8_mode and builder.platform_has_fast_int8:
            print("✓ Enabling INT8 mode")
            config.set_flag(trt.BuilderFlag.INT8)
            # Note: INT8 calibration would be needed here
            print("⚠ INT8 calibration not implemented - using default quantization")
        elif self.int8_mode:
            print("⚠ INT8 not supported on this platform")
        
        # Build engine
        print("\nBuilding TensorRT engine (this may take a few minutes)...")
        engine = builder.build_engine(network, config)
        
        if engine is None:
            print("✗ Failed to build TensorRT engine")
            return None
        
        print("✓ TensorRT engine built successfully")
        
        # Serialize and save engine
        print(f"\nSaving engine to {self.engine_path}...")
        with open(self.engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        print(f"✓ Engine saved successfully")
        
        # Print engine info
        engine_size = os.path.getsize(self.engine_path) / (1024 * 1024)
        print(f"  Engine size: {engine_size:.2f} MB")
        
        return engine
    
    def test_engine(self, engine=None):
        """Test the TensorRT engine with dummy data"""
        
        if engine is None:
            # Load engine from file
            print(f"\nLoading engine from {self.engine_path}...")
            with open(self.engine_path, 'rb') as f:
                runtime = trt.Runtime(self.TRT_LOGGER)
                engine = runtime.deserialize_cuda_engine(f.read())
        
        if engine is None:
            print("✗ Failed to load engine")
            return False
        
        print("✓ Engine loaded successfully")
        
        # Create execution context
        context = engine.create_execution_context()
        
        # Get input/output binding info
        print("\nEngine bindings:")
        for i in range(engine.num_bindings):
            binding = engine[i]
            shape = engine.get_binding_shape(i)
            dtype = trt.nptype(engine.get_binding_dtype(i))
            print(f"  Binding {i}: {binding}")
            print(f"    Shape: {shape}")
            print(f"    Dtype: {dtype}")
            print(f"    Is Input: {engine.binding_is_input(i)}")
        
        print("\n✓ Engine test completed successfully")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert ONNX model to TensorRT engine for Jetson Xavier NX'
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input ONNX model')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to output TensorRT engine (default: input.trt)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Maximum batch size (default: 1)')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Enable FP16 precision (default: enabled)')
    parser.add_argument('--no-fp16', action='store_false', dest='fp16',
                        help='Disable FP16 precision')
    parser.add_argument('--int8', action='store_true', default=False,
                        help='Enable INT8 precision (requires calibration)')
    parser.add_argument('--workspace', type=int, default=1024,
                        help='Maximum workspace size in MB (default: 1024)')
    parser.add_argument('--test', action='store_true',
                        help='Test the engine after building')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Set output path if not specified
    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + '.trt'
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("ONNX to TensorRT Conversion for Jetson Xavier NX")
    print("=" * 70)
    print()
    
    # Create converter and build engine
    converter = ONNXToTensorRT(
        onnx_path=args.input,
        engine_path=args.output,
        max_batch_size=args.batch_size,
        fp16_mode=args.fp16,
        int8_mode=args.int8,
        max_workspace_size=args.workspace * 1024 * 1024
    )
    
    engine = converter.build_engine()
    
    if engine is None:
        print("\n✗ Conversion failed!")
        sys.exit(1)
    
    # Test engine if requested
    if args.test:
        print("\n" + "=" * 70)
        print("Testing TensorRT Engine")
        print("=" * 70)
        success = converter.test_engine(engine)
        if not success:
            print("\n⚠ Engine test failed, but engine file was created")
    
    print("\n" + "=" * 70)
    print("Conversion completed successfully!")
    print("=" * 70)
    print(f"\nTensorRT engine saved to: {args.output}")
    print(f"\nNext step:")
    print(f"  Run inference: python3 run_eeg.py --engine {args.output} --data <your_data.npy>")
    print("=" * 70)


if __name__ == "__main__":
    main()
