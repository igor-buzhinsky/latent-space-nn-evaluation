"""Shared helpers for the tests."""

import os

import torch
from torch.utils.data import TensorDataset

from latentspace.datasets import DatasetWrapper
from latentspace.logging_util import LogUtil
from latentspace.torch_util import to_device


class TensorDatasetWrapper(DatasetWrapper):
    """Dataset wrapper around in-memory tensors, so that tests do not need to download datasets."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, classes: list[str]) -> None:
        """
        Constructs `TensorDatasetWrapper`.

        :param images: Images of shape [N, C, H, W].
        :param labels: Integer labels of shape [N].
        :param classes: Class names.
        """
        super().__init__(images.shape[-1], classes, classes, test_transform=lambda x: x, train_transform=lambda x: x)
        dataset = TensorDataset(to_device(images), labels)
        self.trainset = self.unaugmented_trainset = self.testset = dataset


def random_mnist_like_dataset(no_images: int = 64) -> TensorDatasetWrapper:
    """Random images in [-1, 1] of MNIST size with random labels 0..9."""
    images = torch.rand(no_images, 1, 28, 28) * 2 - 1
    labels = torch.randint(0, 10, (no_images,))
    return TensorDatasetWrapper(images, labels, [str(i) for i in range(10)])


def read_log() -> str:
    """Contents of the current log file."""
    with open(os.path.join(LogUtil._dirname, "log.txt"), encoding="utf-8") as f:
        return f.read()


def number_of_figures(prefix: str = "") -> int:
    """Number of figures in the current log directory, optionally only with the given prefix."""
    return sum(f.startswith("fig_" + prefix) for f in os.listdir(LogUtil._dirname))
