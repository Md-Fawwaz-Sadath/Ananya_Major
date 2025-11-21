#!/usr/bin/env python3
"""
EEG Inference Script for Jetson Xavier NX
Runs EEGNet model inference using TensorRT, ONNX, or Keras (.h5) formats
"""

import os
import sys
import argparse
import numpy as np
import time
from eeg_reduction import eeg_reduction


class EEGInferenceEngine:
    """Unified inference engine supporting TensorRT, ONNX, and Keras models"""
    
    def __init__(self, model_path, model_type='auto'):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to model file (.trt, .onnx, or .h5)
            model_type: Model type ('tensorrt', 'onnx', 'keras', or 'auto')
        """
        self.model_path = model_path
        
        # Auto-detect model type if not specified
        if model_type == 'auto':
            if model_path.endswith('.trt'):
                model_type = 'tensorrt'
            elif model_path.endswith('.onnx'):
                model_type = 'onnx'
            elif model_path.endswith('.h5'):
                model_type = 'keras'
            else:
                raise ValueError(f"Cannot auto-detect model type for: {model_path}")
        
        self.model_type = model_type
        print(f"Initializing {model_type.upper()} inference engine...")
        
        # Load model based on type
        if model_type == 'tensorrt':
            self._init_tensorrt()
        elif model_type == 'onnx':
            self._init_onnx()
        elif model_type == 'keras':
            self._init_keras()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        print("✓ Model loaded successfully")
    
    def _init_tensorrt(self):
        """Initialize TensorRT engine"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError as e:
            print(f"Error: TensorRT libraries not available: {e}")
            print("Make sure you're running on Jetson with JetPack installed")
            sys.exit(1)
        
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        with open(self.model_path, 'rb') as f:
            runtime = trt.Runtime(self.TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def _init_onnx(self):
        """Initialize ONNX runtime"""
        try:
            import onnxruntime as ort
        except ImportError:
            print("Error: onnxruntime not installed")
            print("Install with: pip3 install onnxruntime")
            sys.exit(1)
        
        # Use CUDA execution provider if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"  Input: {self.input_name}, shape={self.session.get_inputs()[0].shape}")
        print(f"  Output: {self.output_name}, shape={self.session.get_outputs()[0].shape}")
    
    def _init_keras(self):
        """Initialize Keras model"""
        try:
            from keras.models import load_model
        except ImportError:
            print("Error: Keras not installed")
            print("Install with: pip3 install keras")
            sys.exit(1)
        
        self.model = load_model(self.model_path)
        print(f"  Input shape: {self.model.input_shape}")
        print(f"  Output shape: {self.model.output_shape}")
    
    def predict(self, input_data):
        """
        Run inference on input data
        
        Args:
            input_data: numpy array with shape matching model input
        
        Returns:
            predictions: numpy array with model predictions
        """
        if self.model_type == 'tensorrt':
            return self._predict_tensorrt(input_data)
        elif self.model_type == 'onnx':
            return self._predict_onnx(input_data)
        elif self.model_type == 'keras':
            return self._predict_keras(input_data)
    
    def _predict_tensorrt(self, input_data):
        """TensorRT inference"""
        import pycuda.driver as cuda
        
        # Copy input data to host buffer
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        # Transfer input data to device
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Transfer predictions back to host
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        
        # Synchronize stream
        self.stream.synchronize()
        
        return self.outputs[0]['host']
    
    def _predict_onnx(self, input_data):
        """ONNX runtime inference"""
        result = self.session.run([self.output_name], {self.input_name: input_data})
        return result[0]
    
    def _predict_keras(self, input_data):
        """Keras inference"""
        return self.model.predict(input_data)


def load_eeg_data(data_path, file_format='auto'):
    """
    Load EEG data from file
    
    Args:
        data_path: Path to data file
        file_format: Format of data ('npy', 'npz', 'csv', or 'auto')
    
    Returns:
        data: numpy array with EEG data
    """
    if file_format == 'auto':
        if data_path.endswith('.npy'):
            file_format = 'npy'
        elif data_path.endswith('.npz'):
            file_format = 'npz'
        elif data_path.endswith('.csv'):
            file_format = 'csv'
        else:
            raise ValueError(f"Cannot auto-detect file format for: {data_path}")
    
    print(f"Loading EEG data from: {data_path} (format: {file_format})")
    
    if file_format == 'npy':
        data = np.load(data_path)
    elif file_format == 'npz':
        npz_file = np.load(data_path)
        # Try common key names
        if 'X' in npz_file:
            data = npz_file['X']
        elif 'data' in npz_file:
            data = npz_file['data']
        elif 'X_test' in npz_file:
            data = npz_file['X_test']
        else:
            # Use first array
            key = list(npz_file.keys())[0]
            data = npz_file[key]
            print(f"  Using key: {key}")
    elif file_format == 'csv':
        data = np.loadtxt(data_path, delimiter=',')
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    print(f"  Data shape: {data.shape}")
    print(f"  Data dtype: {data.dtype}")
    
    return data


def preprocess_data(data, n_ds=1, n_ch=64, T=3, fs=160):
    """
    Preprocess EEG data
    
    Args:
        data: Raw EEG data
        n_ds: Downsampling factor
        n_ch: Number of channels to use
        T: Time window in seconds
        fs: Sampling frequency in Hz
    
    Returns:
        Preprocessed data ready for model input
    """
    print(f"Preprocessing EEG data...")
    print(f"  Downsampling: {n_ds}x")
    print(f"  Channels: {n_ch}")
    print(f"  Time window: {T}s")
    
    # Apply EEG reduction
    if len(data.shape) == 2:
        # Add trial dimension if needed
        data = np.expand_dims(data, axis=0)
    
    # Apply reduction
    data_reduced = eeg_reduction(data, n_ds=n_ds, n_ch=n_ch, T=T, fs=fs)
    
    # Add channel dimension for model
    data_reduced = np.expand_dims(data_reduced, axis=-1)
    
    print(f"  Preprocessed shape: {data_reduced.shape}")
    
    return data_reduced


def main():
    parser = argparse.ArgumentParser(
        description='Run EEGNet inference on Jetson Xavier NX'
    )
    
    # Model arguments
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to model file (.trt, .onnx, or .h5)')
    parser.add_argument('--model-type', type=str, default='auto',
                        choices=['auto', 'tensorrt', 'onnx', 'keras'],
                        help='Model type (default: auto-detect)')
    
    # Data arguments
    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Path to input EEG data (.npy, .npz, or .csv)')
    parser.add_argument('--data-format', type=str, default='auto',
                        choices=['auto', 'npy', 'npz', 'csv'],
                        help='Input data format (default: auto-detect)')
    
    # Preprocessing arguments
    parser.add_argument('--channels', type=int, default=64,
                        help='Number of EEG channels (default: 64)')
    parser.add_argument('--downsample', type=int, default=1,
                        help='Downsampling factor (default: 1)')
    parser.add_argument('--time-window', type=float, default=3.0,
                        help='Time window in seconds (default: 3.0)')
    parser.add_argument('--sampling-rate', type=int, default=160,
                        help='Sampling rate in Hz (default: 160)')
    
    # Output arguments
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Save predictions to file (optional)')
    parser.add_argument('--num-classes', type=int, default=4,
                        help='Number of output classes (default: 4)')
    
    # Performance arguments
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark to measure inference time')
    parser.add_argument('--num-iterations', type=int, default=100,
                        help='Number of iterations for benchmarking (default: 100)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)
    
    print("=" * 70)
    print("EEGNet Inference on Jetson Xavier NX")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print("=" * 70)
    print()
    
    # Initialize inference engine
    engine = EEGInferenceEngine(args.model, args.model_type)
    print()
    
    # Load data
    data = load_eeg_data(args.data, args.data_format)
    print()
    
    # Preprocess data
    data_preprocessed = preprocess_data(
        data,
        n_ds=args.downsample,
        n_ch=args.channels,
        T=args.time_window,
        fs=args.sampling_rate
    )
    print()
    
    # Run inference
    print("Running inference...")
    start_time = time.time()
    predictions = engine.predict(data_preprocessed)
    inference_time = (time.time() - start_time) * 1000
    
    print(f"✓ Inference completed in {inference_time:.2f} ms")
    print()
    
    # Process predictions
    print("=" * 70)
    print("Predictions")
    print("=" * 70)
    
    # Reshape predictions if needed
    predictions = np.array(predictions).reshape(-1, args.num_classes)
    
    class_names = [f"Class {i}" for i in range(args.num_classes)]
    
    for i, pred in enumerate(predictions):
        predicted_class = np.argmax(pred)
        confidence = pred[predicted_class] * 100
        
        print(f"\nSample {i+1}:")
        print(f"  Predicted class: {predicted_class} ({class_names[predicted_class]})")
        print(f"  Confidence: {confidence:.2f}%")
        print(f"  Class probabilities:")
        for j, prob in enumerate(pred):
            print(f"    {class_names[j]}: {prob*100:.2f}%")
    
    # Save predictions if requested
    if args.output:
        np.save(args.output, predictions)
        print(f"\n✓ Predictions saved to: {args.output}")
    
    # Run benchmark if requested
    if args.benchmark:
        print("\n" + "=" * 70)
        print(f"Benchmarking ({args.num_iterations} iterations)")
        print("=" * 70)
        
        times = []
        for i in range(args.num_iterations):
            start = time.time()
            _ = engine.predict(data_preprocessed)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                print(f"  Iteration {i+1}/{args.num_iterations}: {elapsed:.2f} ms")
        
        times = np.array(times)
        print(f"\nBenchmark Results:")
        print(f"  Mean: {times.mean():.2f} ms")
        print(f"  Std: {times.std():.2f} ms")
        print(f"  Min: {times.min():.2f} ms")
        print(f"  Max: {times.max():.2f} ms")
        print(f"  Throughput: {1000/times.mean():.2f} samples/sec")
    
    print("\n" + "=" * 70)
    print("Inference completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
