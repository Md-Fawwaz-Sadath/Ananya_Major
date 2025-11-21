#!/usr/bin/env python3
#*----------------------------------------------------------------------------*
#* Real-time EEG inference helper for Jetson Xavier NX                        *
#*----------------------------------------------------------------------------*

"""
Real-time EEG inference helper
This module provides utilities for continuous real-time EEG inference
"""

import numpy as np
import time
import threading
from collections import deque
from inference import EEGNetInference


class RealTimeEEGInference:
    """Real-time EEG inference with buffering and continuous processing"""
    
    def __init__(self, model_path, num_classes=4, n_ds=1, n_ch=64, T=3, fs=160):
        """
        Initialize real-time inference engine
        
        Args:
            model_path: Path to trained model
            num_classes: Number of classes
            n_ds: Downsampling factor
            n_ch: Number of channels
            T: Time window in seconds
            fs: Sampling frequency in Hz
        """
        self.fs = fs
        self.T = T
        self.n_ch = n_ch
        self.n_samples = int(T * fs)
        
        # Initialize inference engine
        self.engine = EEGNetInference(
            model_path=model_path,
            num_classes=num_classes,
            n_ds=n_ds,
            n_ch=n_ch,
            T=T
        )
        
        # Data buffer
        self.buffer = deque(maxlen=self.n_samples)
        self.buffer_lock = threading.Lock()
        
        # Processing state
        self.is_running = False
        self.processing_thread = None
        self.callback = None
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'avg_inference_time': 0.0,
            'last_prediction': None
        }
    
    def add_sample(self, sample):
        """
        Add a new EEG sample to the buffer
        
        Args:
            sample: EEG data for one time point, shape (n_channels,)
        """
        if sample.shape[0] != self.n_ch:
            raise ValueError(f"Expected {self.n_ch} channels, got {sample.shape[0]}")
        
        with self.buffer_lock:
            self.buffer.append(sample)
    
    def add_batch(self, batch):
        """
        Add a batch of samples to the buffer
        
        Args:
            batch: EEG data, shape (n_samples, n_channels) or (n_channels, n_samples)
        """
        if len(batch.shape) == 2:
            if batch.shape[0] == self.n_ch:
                # Shape: (n_channels, n_samples)
                batch = batch.T  # Transpose to (n_samples, n_channels)
            
            for sample in batch:
                self.add_sample(sample)
        else:
            raise ValueError(f"Invalid batch shape: {batch.shape}")
    
    def get_buffer(self):
        """Get current buffer contents"""
        with self.buffer_lock:
            if len(self.buffer) < self.n_samples:
                return None
            return np.array(list(self.buffer))
    
    def is_ready(self):
        """Check if buffer has enough data for inference"""
        return len(self.buffer) >= self.n_samples
    
    def predict_from_buffer(self):
        """
        Run inference on current buffer
        
        Returns:
            dict with prediction results or None if buffer not ready
        """
        buffer_data = self.get_buffer()
        if buffer_data is None:
            return None
        
        # Reshape to (n_channels, n_samples)
        eeg_data = buffer_data.T
        
        # Run inference
        result = self.engine.predict_single(eeg_data)
        
        # Update statistics
        self.stats['total_predictions'] += 1
        n = self.stats['total_predictions']
        self.stats['avg_inference_time'] = (
            (n - 1) * self.stats['avg_inference_time'] + result['inference_time_ms']
        ) / n
        self.stats['last_prediction'] = result
        
        return result
    
    def set_callback(self, callback_func):
        """
        Set callback function to be called after each prediction
        
        Args:
            callback_func: Function that takes prediction result as argument
        """
        self.callback = callback_func
    
    def _processing_loop(self, interval=0.1):
        """Internal processing loop"""
        while self.is_running:
            if self.is_ready():
                result = self.predict_from_buffer()
                if result and self.callback:
                    try:
                        self.callback(result)
                    except Exception as e:
                        print(f"Error in callback: {e}")
            time.sleep(interval)
    
    def start(self, interval=0.1):
        """
        Start continuous inference processing
        
        Args:
            interval: Time between inference attempts in seconds
        """
        if self.is_running:
            print("Already running!")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            args=(interval,),
            daemon=True
        )
        self.processing_thread.start()
        print("Real-time inference started!")
    
    def stop(self):
        """Stop continuous inference processing"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        print("Real-time inference stopped!")
    
    def get_stats(self):
        """Get inference statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_predictions': 0,
            'avg_inference_time': 0.0,
            'last_prediction': None
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.stop()
        self.engine.cleanup()


class EEGDataSimulator:
    """Simulate EEG data for testing"""
    
    def __init__(self, n_ch=64, fs=160, duration=10.0):
        """
        Initialize EEG data simulator
        
        Args:
            n_ch: Number of channels
            fs: Sampling frequency
            duration: Duration of simulation in seconds
        """
        self.n_ch = n_ch
        self.fs = fs
        self.duration = duration
        self.t = 0.0
        self.dt = 1.0 / fs
    
    def generate_sample(self):
        """Generate a single sample of simulated EEG data"""
        # Simple sinusoidal simulation with noise
        sample = np.zeros(self.n_ch)
        for i in range(self.n_ch):
            freq = 10 + i * 0.5  # Different frequency per channel
            sample[i] = np.sin(2 * np.pi * freq * self.t) + 0.1 * np.random.randn()
        
        self.t += self.dt
        return sample
    
    def generate_batch(self, n_samples):
        """Generate a batch of samples"""
        batch = np.zeros((n_samples, self.n_ch))
        for i in range(n_samples):
            batch[i] = self.generate_sample()
        return batch.T  # Return as (n_channels, n_samples)


def example_usage():
    """Example usage of real-time inference"""
    
    # Initialize inference engine
    model_path = "models/global_class_4_ds1_nch64_T3_split_0.h5"
    rt_inference = RealTimeEEGInference(
        model_path=model_path,
        num_classes=4,
        n_ch=64,
        T=3.0,
        fs=160
    )
    
    # Define callback function
    def prediction_callback(result):
        print(f"Class: {result['class']}, "
              f"Confidence: {result['confidence']:.2f}, "
              f"Time: {result['inference_time_ms']:.2f}ms")
    
    rt_inference.set_callback(prediction_callback)
    
    # Start real-time processing
    rt_inference.start(interval=0.1)  # Check every 100ms
    
    # Simulate data acquisition
    simulator = EEGDataSimulator(n_ch=64, fs=160)
    
    try:
        print("Simulating EEG data... (Press Ctrl+C to stop)")
        for _ in range(1000):  # Simulate 1000 samples
            sample = simulator.generate_sample()
            rt_inference.add_sample(sample)
            time.sleep(1.0 / 160)  # Simulate 160 Hz sampling
            
            # Print stats every 100 samples
            if rt_inference.stats['total_predictions'] % 10 == 0 and rt_inference.stats['total_predictions'] > 0:
                stats = rt_inference.get_stats()
                print(f"\nStats: {stats['total_predictions']} predictions, "
                      f"avg time: {stats['avg_inference_time']:.2f}ms\n")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        rt_inference.stop()
        stats = rt_inference.get_stats()
        print(f"\nFinal stats: {stats}")
        rt_inference.cleanup()


if __name__ == '__main__':
    example_usage()
