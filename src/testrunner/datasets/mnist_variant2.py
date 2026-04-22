# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Download, convert, and cache the pneumonia_mnist dataset as float64 .bin files.

The underlying source is PneumoniaMNIST from MedMNIST v3 (Yang et al., 2023).
Images are already grayscale (single channel).

Samples are truncated so the train and test splits divide evenly by the
conv-MLP eval_batch_size of 624:
    train = shuffle(train + val)[:4992]   (from 4708 + 524 = 5232)
    test  = test[:624]                     (from 624, no change)
"""

import os
import urllib.request
from pathlib import Path

import numpy as np


_URL = "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1"
_NPZ_NAME = "source.npz"
_NUM_CLASSES = 2
_IMAGE_SIZE = 28 * 28
_TRAIN_N = 4992
_TEST_N = 624
_SHUFFLE_SEED = 0


def prepare_mnist_variant2():
    """Prepare the pneumonia_mnist dataset, returning paths to cached .bin files."""
    cache_dir = _get_cache_dir() / "pneumonia_mnist"
    raw_file = cache_dir / _NPZ_NAME

    if not (cache_dir / "train_images.bin").exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not raw_file.exists():
            print(f"Downloading {_NPZ_NAME}...")
            urllib.request.urlretrieve(_URL, raw_file)
            print(f"Downloaded {_NPZ_NAME}")
        _convert_to_bin(raw_file, cache_dir)

    return {
        "train_images": str(cache_dir / "train_images.bin"),
        "train_labels": str(cache_dir / "train_labels.bin"),
        "test_images": str(cache_dir / "test_images.bin"),
        "test_labels": str(cache_dir / "test_labels.bin"),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_cache_dir():
    return Path(
        os.environ.get(
            "MININNVERIFIER_CACHE_DIR", Path.home() / ".cache" / "mininnverifier" / "datasets"
        )
    )


def _images_to_float(gray):
    """Grayscale (N, 28, 28) uint8 -> float64 (N, 784) in [0, 1]."""
    return gray.astype(np.float64).reshape(gray.shape[0], _IMAGE_SIZE) / 255.0


def _labels_to_one_hot(labels):
    """Labels (N, 1) int in {0, 1} -> one-hot float64 (N, 2)."""
    labels = labels.ravel().astype(np.int64)
    one_hot = np.zeros((len(labels), _NUM_CLASSES), dtype=np.float64)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


def _convert_to_bin(npz_path, out_dir):
    data = np.load(npz_path)

    train_images = np.concatenate([data["train_images"], data["val_images"]])
    train_labels = np.concatenate([data["train_labels"], data["val_labels"]])
    test_images = data["test_images"]
    test_labels = data["test_labels"]

    rng = np.random.default_rng(_SHUFFLE_SEED)
    perm = rng.permutation(len(train_images))
    train_images = train_images[perm][:_TRAIN_N]
    train_labels = train_labels[perm][:_TRAIN_N]
    test_images = test_images[:_TEST_N]
    test_labels = test_labels[:_TEST_N]

    _images_to_float(train_images).tofile(out_dir / "train_images.bin")
    _labels_to_one_hot(train_labels).tofile(out_dir / "train_labels.bin")
    _images_to_float(test_images).tofile(out_dir / "test_images.bin")
    _labels_to_one_hot(test_labels).tofile(out_dir / "test_labels.bin")
