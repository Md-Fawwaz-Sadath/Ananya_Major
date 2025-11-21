#!/usr/bin/env python3
"""
Create test EEG data for inference testing
"""

import numpy as np
import argparse

def create_test_data(output_path, num_trials=5, num_channels=64, num_samples=480, seed=42):
    """
    Create synthetic EEG test data
    
    Args:
        output_path: Path to save the data
        num_trials: Number of trials/samples
        num_channels: Number of EEG channels
        num_samples: Number of time samples (480 = 3 seconds at 160 Hz)
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    print("=" * 70)
    print("Creating Test EEG Data")
    print("=" * 70)
    print(f"  Trials: {num_trials}")
    print(f"  Channels: {num_channels}")
    print(f"  Samples: {num_samples}")
    print(f"  Shape: ({num_trials}, {num_channels}, {num_samples})")
    print(f"  Output: {output_path}")
    print()
    
    # Create synthetic EEG data
    # Using random normal distribution to simulate EEG signals
    data = np.random.randn(num_trials, num_channels, num_samples).astype(np.float32)
    
    # Optional: Add some structure to make it more realistic
    # Add slow oscillations (theta/alpha bands)
    t = np.linspace(0, 3, num_samples)
    for i in range(num_trials):
        for ch in range(num_channels):
            # Add 8-12 Hz oscillations (alpha)
            freq = np.random.uniform(8, 12)
            data[i, ch, :] += 0.5 * np.sin(2 * np.pi * freq * t)
            
            # Add 4-8 Hz oscillations (theta)
            freq = np.random.uniform(4, 8)
            data[i, ch, :] += 0.3 * np.sin(2 * np.pi * freq * t)
    
    # Save data
    np.save(output_path, data)
    
    print("✓ Test data created successfully")
    print()
    print(f"Data statistics:")
    print(f"  Mean: {data.mean():.4f}")
    print(f"  Std: {data.std():.4f}")
    print(f"  Min: {data.min():.4f}")
    print(f"  Max: {data.max():.4f}")
    print(f"  Size: {data.nbytes / (1024*1024):.2f} MB")
    print()
    print("=" * 70)
    print("Next steps:")
    print("=" * 70)
    print(f"1. Run inference:")
    print(f"   python3 run_eeg.py --model model.trt --data {output_path} --num-classes 4")
    print()
    print(f"2. Or with ONNX model:")
    print(f"   python3 run_eeg.py --model output.onnx --data {output_path} --num-classes 4")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Create synthetic EEG test data')
    parser.add_argument('--output', '-o', type=str, default='test_data.npy',
                        help='Output file path (default: test_data.npy)')
    parser.add_argument('--trials', type=int, default=5,
                        help='Number of trials (default: 5)')
    parser.add_argument('--channels', type=int, default=64,
                        help='Number of channels (default: 64)')
    parser.add_argument('--samples', type=int, default=480,
                        help='Number of samples (default: 480)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    create_test_data(
        output_path=args.output,
        num_trials=args.trials,
        num_channels=args.channels,
        num_samples=args.samples,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
