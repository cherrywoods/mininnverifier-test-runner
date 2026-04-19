# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Download, convert, and cache the mnist_variant dataset as float64 .bin files.

The underlying source is DermaMNIST from MedMNIST v3 (Yang et al., 2023).
Color images are reduced to a single grayscale channel by taking the red
channel, which carries the majority of useful signal for this source data
while matching the 1-channel shape of MNIST.

The source has 7 classes which are merged down to 5 by iteratively merging
the two smallest classes until 5 remain. The resulting mapping is::

    source class 0, 3, 6 -> 0
    source class 1        -> 1
    source class 2        -> 2
    source class 4        -> 3
    source class 5        -> 4

Samples are truncated so the train and test splits divide evenly by the
MLP/conv-MLP eval_batch_size of 1000:
    train = shuffle(train + val)[:8000]   (from 7007 + 1003 = 8010)
    test  = test[:2000]                    (from 2005)
"""

import os
import urllib.request
from pathlib import Path

import numpy as np


_URL = "https://zenodo.org/records/10519652/files/dermamnist.npz?download=1"
_NPZ_NAME = "source.npz"
_NUM_CLASSES = 5
_IMAGE_SIZE = 28 * 28
# Maps each source class to its merged target class (see module docstring).
_CLASS_MAP = np.array([0, 1, 2, 0, 3, 4, 0], dtype=np.int64)
_TRAIN_N = 8000
_TEST_N = 2000
_SHUFFLE_SEED = 0


def prepare_mnist_variant():
    """Prepare the mnist_variant dataset, returning paths to cached .bin files."""
    cache_dir = _get_cache_dir() / "mnist_variant"
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
            "MININNVERIFIER_CACHE_DIR",
            Path.home() / ".cache" / "mininnverifier" / "datasets",
        )
    )


def _images_to_grayscale(rgb):
    """RGB (N, 28, 28, 3) uint8 -> grayscale float64 (N, 784) in [0, 1].

    Uses the red channel only.
    """
    red = rgb[..., 0].astype(np.float64) / 255.0
    return red.reshape(rgb.shape[0], _IMAGE_SIZE)


def _labels_to_one_hot(labels):
    """Labels (N, 1) int in [0, 7) -> merged one-hot float64 (N, 5)."""
    labels = _CLASS_MAP[labels.ravel().astype(np.int64)]
    one_hot = np.zeros((len(labels), _NUM_CLASSES), dtype=np.float64)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


def _convert_to_bin(npz_path, out_dir):
    data = np.load(npz_path)

    train_images = np.concatenate([data["train_images"], data["val_images"]])
    train_labels = np.concatenate([data["train_labels"], data["val_labels"]])
    test_images = data["test_images"]
    test_labels = data["test_labels"]

    # Shuffle the combined train+val so validation samples are not clustered
    # at the end, then truncate to a size divisible by eval_batch_size.
    rng = np.random.default_rng(_SHUFFLE_SEED)
    perm = rng.permutation(len(train_images))
    train_images = train_images[perm][:_TRAIN_N]
    train_labels = train_labels[perm][:_TRAIN_N]
    test_images = test_images[:_TEST_N]
    test_labels = test_labels[:_TEST_N]

    train_x = _images_to_grayscale(train_images)
    train_y = _labels_to_one_hot(train_labels)
    test_x = _images_to_grayscale(test_images)
    test_y = _labels_to_one_hot(test_labels)

    train_x.tofile(out_dir / "train_images.bin")
    train_y.tofile(out_dir / "train_labels.bin")
    test_x.tofile(out_dir / "test_images.bin")
    test_y.tofile(out_dir / "test_labels.bin")
