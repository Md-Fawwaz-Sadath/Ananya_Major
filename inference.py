#!/usr/bin/env python3
"""
Real-time / batch inference helper for the EEGNet-based BCI models.

Designed for deployment on NVIDIA Jetson platforms (e.g., Xavier NX)
running TensorFlow/Keras. The script loads a trained `.h5` model,
preprocesses incoming EEG data with the same downsampling, channel
selection, and time-windowing used during training, and reports the
predicted class probabilities.
"""
import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np

from eeg_reduction import eeg_reduction

try:
    from keras.models import load_model
    from keras import backend as K
except ImportError as exc:
    print("Missing Keras/TensorFlow dependencies. "
          "Make sure they are installed on the Jetson target.", file=sys.stderr)
    raise exc


def configure_gpu(memory_growth: bool = True) -> None:
    """
    Enable on-demand GPU memory allocation. Works with TF 1.x (JetPack 4.x)
    and TF 2.x (JetPack 5.x).
    """
    try:
        import tensorflow as tf
    except ImportError:
        return

    try:
        version_major = int(tf.__version__.split(".")[0])
    except (ValueError, AttributeError):
        version_major = 1

    if version_major >= 2:
        if memory_growth:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    else:
        if memory_growth:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.Session(config=config)
            K.set_session(session)


def load_array(path: str, npz_key: Optional[str] = None) -> np.ndarray:
    """
    Load EEG data from .npy or .npz.

    Expected shape before preprocessing:
        (n_trials, n_channels, n_samples)
    or  (n_channels, n_samples) for a single trial.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input array {path} was not found.")

    if path.endswith(".npz"):
        with np.load(path) as data:
            if npz_key is None:
                raise ValueError("When loading .npz files you must provide --npz-key.")
            if npz_key not in data:
                raise KeyError(f"Key '{npz_key}' not found in {path}. "
                               f"Available keys: {list(data.keys())}")
            arr = data[npz_key]
    else:
        arr = np.load(path)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array after preprocessing, got shape {arr.shape}")
    return arr


def normalize(
    x: np.ndarray,
    axis: int = -1,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Channel-wise z-score normalization.
    """
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True) + eps
    return (x - mean) / std


def prepare_input(
    raw: np.ndarray,
    n_ds: int,
    n_ch: int,
    window_s: float,
    sample_rate: int,
    apply_reduction: bool,
    do_normalize: bool,
) -> np.ndarray:
    """
    Apply the same preprocessing used for training and add EEGNet channel dim.
    """
    data = raw
    if apply_reduction:
        data = eeg_reduction(
            data,
            n_ds=n_ds,
            n_ch=n_ch,
            T=window_s,
            fs=sample_rate,
        )
    if do_normalize:
        data = normalize(data, axis=-1)
    data = np.expand_dims(data, axis=-1)
    return data


def run_inference(
    model_path: str,
    data: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """
    Load the Keras model and run inference.
    """
    print(f"[INFO] Loading model from {model_path}")
    model = load_model(model_path)
    print(model.summary())

    print(f"[INFO] Running inference on {len(data)} trial(s)")
    start = time.time()
    probs = model.predict(data, batch_size=batch_size, verbose=0)
    latency = (time.time() - start) / len(data)
    print(f"[INFO] Mean per-trial latency: {latency * 1e3:.2f} ms")
    return probs


def save_outputs(
    probs: np.ndarray,
    path: Optional[str],
    class_map: Optional[str],
) -> None:
    """
    Persist probabilities and (optional) decoded labels.
    """
    if path is None:
        return

    payload = {"probabilities": probs.tolist()}
    if class_map:
        with open(class_map, "r", encoding="utf-8") as handle:
            labels = json.load(handle)
        payload["classes"] = labels
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[INFO] Saved predictions to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EEGNet inference helper for Jetson Xavier NX."
    )
    parser.add_argument("--model", required=True, help="Path to trained .h5 model.")
    parser.add_argument("--data", required=True, help="Path to .npy/.npz EEG tensor.")
    parser.add_argument("--npz-key", help="Key inside .npz file that holds the EEG tensor.")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--downsample", type=int, default=1, help="Downsampling factor used during training.")
    parser.add_argument("--channels", type=int, default=64, help="Number of EEG channels to keep.")
    parser.add_argument("--window", type=float, default=3.0, help="Window duration in seconds.")
    parser.add_argument("--sample-rate", type=int, default=160, help="Original sampling rate in Hz.")
    parser.add_argument("--no-reduction", action="store_true", help="Skip eeg_reduction preprocessing.")
    parser.add_argument("--normalize", action="store_true", help="Apply per-channel z-score before inference.")
    parser.add_argument("--save-json", help="Optional path to dump probabilities as JSON.")
    parser.add_argument("--class-map", help="JSON file mapping class indices to labels.")
    parser.add_argument("--gpu-memory-growth", action="store_true", default=True,
                        help="Enable on-demand GPU memory allocation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.gpu_memory_growth:
        configure_gpu(memory_growth=True)

    raw = load_array(args.data, args.npz_key)
    data = prepare_input(
        raw,
        n_ds=args.downsample,
        n_ch=args.channels,
        window_s=args.window,
        sample_rate=args.sample_rate,
        apply_reduction=not args.no_reduction,
        do_normalize=args.normalize,
    )

    probs = run_inference(
        model_path=args.model,
        data=data,
        batch_size=args.batch_size,
    )

    top_classes = probs.argmax(axis=1)
    for idx, (cls, prob) in enumerate(zip(top_classes, probs)):
        print(f"[RESULT] Trial {idx}: class={cls} prob={prob[cls]:.4f} full={prob}")

    save_outputs(probs, args.save_json, args.class_map)


if __name__ == "__main__":
    main()
