#!/usr/bin/env python3
"""
Test script to verify ONNX model is valid and check its properties
"""

import sys
import os

def test_onnx_model(model_path):
    """Test and display information about ONNX model"""
    
    print("=" * 70)
    print("ONNX Model Verification")
    print("=" * 70)
    print(f"Model: {model_path}")
    print()
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"✗ Error: Model file not found: {model_path}")
        return False
    
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✓ File exists")
    print(f"  Size: {file_size:.2f} MB")
    print()
    
    # Try to load with ONNX
    try:
        import onnx
        print("Loading model with ONNX...")
        model = onnx.load(model_path)
        print("✓ Model loaded successfully")
        print()
        
        # Check model
        print("Validating model...")
        try:
            onnx.checker.check_model(model)
            print("✓ Model validation passed")
        except Exception as e:
            print(f"⚠ Model validation warning: {e}")
        print()
        
        # Display model info
        print("Model Information:")
        print(f"  IR Version: {model.ir_version}")
        print(f"  Producer: {model.producer_name} {model.producer_version}")
        print(f"  Domain: {model.domain}")
        print(f"  Model Version: {model.model_version}")
        print(f"  Doc String: {model.doc_string}")
        print()
        
        # Display graph info
        graph = model.graph
        print(f"Graph: {graph.name}")
        print(f"  Nodes: {len(graph.node)}")
        print(f"  Initializers: {len(graph.initializer)}")
        print()
        
        # Display inputs
        print("Model Inputs:")
        for i, input_tensor in enumerate(graph.input):
            # Skip initializers
            if input_tensor.name in [init.name for init in graph.initializer]:
                continue
            
            shape = []
            for dim in input_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append('?')
            
            dtype = onnx.TensorProto.DataType.Name(input_tensor.type.tensor_type.elem_type)
            print(f"  [{i}] {input_tensor.name}")
            print(f"      Shape: {shape}")
            print(f"      Type: {dtype}")
        print()
        
        # Display outputs
        print("Model Outputs:")
        for i, output_tensor in enumerate(graph.output):
            shape = []
            for dim in output_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append('?')
            
            dtype = onnx.TensorProto.DataType.Name(output_tensor.type.tensor_type.elem_type)
            print(f"  [{i}] {output_tensor.name}")
            print(f"      Shape: {shape}")
            print(f"      Type: {dtype}")
        print()
        
    except ImportError:
        print("⚠ ONNX library not installed. Install with: pip3 install onnx")
        print("  Skipping detailed model inspection...")
        print()
    except Exception as e:
        print(f"✗ Error loading model with ONNX: {e}")
        return False
    
    # Try to load with ONNX Runtime
    try:
        import onnxruntime as ort
        print("Testing with ONNX Runtime...")
        
        # Try CPU provider first
        try:
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            print("✓ Model loaded in ONNX Runtime (CPU)")
            
            # Display provider info
            print(f"  Available providers: {ort.get_available_providers()}")
            print()
            
            # Test inference with dummy data
            import numpy as np
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            
            # Create dummy input
            # Replace dynamic dimensions with fixed values
            test_shape = []
            for dim in input_shape:
                if isinstance(dim, str) or dim is None:
                    test_shape.append(1)
                else:
                    test_shape.append(dim)
            
            print(f"Running test inference...")
            print(f"  Input shape: {test_shape}")
            dummy_input = np.random.randn(*test_shape).astype(np.float32)
            
            result = session.run(None, {input_name: dummy_input})
            print(f"✓ Test inference successful")
            print(f"  Output shape: {result[0].shape}")
            print()
            
        except Exception as e:
            print(f"⚠ ONNX Runtime test failed: {e}")
            print()
        
    except ImportError:
        print("⚠ ONNX Runtime not installed. Install with: pip3 install onnxruntime")
        print()
    except Exception as e:
        print(f"⚠ ONNX Runtime error: {e}")
        print()
    
    print("=" * 70)
    print("Verification Complete")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. If on Jetson: Convert to TensorRT with:")
    print(f"   python3 onnx_to_tensorrt.py --input {model_path}")
    print()
    print("2. Run inference with:")
    print(f"   python3 run_eeg.py --model {model_path} --data your_data.npy")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_onnx_model.py <path_to_onnx_model>")
        print()
        print("Example: python3 test_onnx_model.py output.onnx")
        sys.exit(1)
    
    model_path = sys.argv[1]
    success = test_onnx_model(model_path)
    
    sys.exit(0 if success else 1)
