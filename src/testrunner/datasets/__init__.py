# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
from .mnist import prepare_mnist
from .mnist_variant import prepare_mnist_variant
from .mnist_variant2 import prepare_mnist_variant2

DATASETS = {
    "mnist": prepare_mnist,
    "mnist_variant": prepare_mnist_variant,
    "mnist_variant2": prepare_mnist_variant2,
}


def prepare_dataset(source_dataset):
    """Download/prepare a dataset and return absolute paths to .bin files.

    source_dataset is a dataset name string (e.g. "mnist").
    """
    if source_dataset not in DATASETS:
        available = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset: {source_dataset!r}. Available: {available}")
    return DATASETS[source_dataset]()
