"""Showing and saving images, and helpers for plots."""

from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.nonparametric.bandwidths
import torch
import torchvision

from latentspace.logging_util import LogUtil
from latentspace.torch_util import to_numpy


def imshow(img: torch.Tensor, figsize: tuple[float, float] = (12, 2)) -> None:
    """
    Shows the image and saves the produced figure.

    :param img: Image to show (PyTorch tensor).
    :param figsize: Matplotlib `figsize`.
    """
    plt.figure(figsize=figsize)
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(to_numpy(img), (1, 2, 0)))
    plt.axis("off")
    LogUtil.savefig("imshow")
    plt.show()
    plt.close()


def imshow_tensors(
    *tensors: torch.Tensor,
    clamp: bool = True,
    figsize: tuple[float, float] = (12, 2),
    captions: list[str] | None = None,
    nrow: int = 8,
    pad_value: float = 1,
) -> None:
    """
    Enhanced `torchvision.utils.make_grid()`. Supports image captions. Also saves the produced figure.

    :param tensors: Images to show.
    :param clamp: If `True`, clamp all pixels to [-1, 1].
    :param figsize: Matplotlib `figsize`.
    :param captions: List of string captions to be printed on top of images.
    :param nrow: `nrow` to be passed to `torchvision.utils.make_grid`.
    :param pad_value: `pad_value` to be passed to `torchvision.utils.make_grid`.
    """
    t = torch.cat(tensors)
    if len(t.shape) != 4:
        raise ValueError(f"Invalid shape of tensors {t.shape}")
    # handling 1D images
    if t.shape[1] == 1:
        t = t.repeat(1, 3, 1, 1)
    if clamp:
        t = torch.clamp(t, -1, 1)
    images: list[Any] = list(t)

    # adding textual captions if they are given (this involves rescaling)
    if captions is not None:

        def multiline_puttext(img: np.ndarray, caption: str) -> np.ndarray:
            """`cv2.putText` does not support line breaks on its own."""
            scale = 0.8
            y0 = img.shape[0] * 0.15
            dy = img.shape[0] * 0.20
            for i, text in enumerate(caption.split("\n")):
                y = int(y0 + i * dy)
                # green
                img = cv2.putText(img, text, (0, y), cv2.FONT_HERSHEY_TRIPLEX, scale, (0, 250, 0))
            return img

        default_size = (128,) * 2
        for i in range(len(images)):
            if not isinstance(captions[i], str):
                raise TypeError("Captions must be str")
            images[i] = (to_numpy(images[i]).transpose(1, 2, 0) / 2 + 0.5) * 255
            # shape = H*W*3
            images[i] = cv2.resize(images[i], default_size, interpolation=cv2.INTER_NEAREST)
            images[i] = multiline_puttext(images[i], captions[i]) / 255
            images[i] = torch.FloatTensor(images[i].transpose(2, 0, 1) * 2 - 1)

    imshow(torchvision.utils.make_grid(torch.stack(images), nrow=nrow, pad_value=pad_value), figsize)


def get_kde_bandwidth(x: np.ndarray) -> float:
    """
    This fixes a peculiar problem with `sns.kdeplot`/`sns.distplot`. Use this to compute the bandwidth and provide it as
    argument `bw`.
    https://stackoverflow.com/questions/61440184/who-is-scott-valueerror-in-seaborn-pairplot-could-not-convert-string-to-floa

    :param x: Input (NumPy array).
    :return: KDE bandwidth computed by Scott's method.
    """
    return statsmodels.nonparametric.bandwidths.bw_scott(x)


class ImageSet:
    """Accumulates images and captions. Shows them in blocks of size `max_num`."""

    def __init__(self, max_num: int) -> None:
        """
        Constructs `ImageSet`.

        :param max_num: Number of images and captions to accumulate until they can be shown.
        """
        self.images, self.captions = [], []
        self.max_num = max_num

    def append(self, images: list[torch.Tensor], captions: list[str] | None = None) -> None:
        """
        Appends the given images and captions.

        :param images: List of images (PyTorch tensors).
        :param captions: List of string captions corresponding to the images.
        """
        self.images += [images]
        if captions is None:
            captions = [""] * len(images)
        self.captions += [captions]

    def maybe_show(self, force: bool = False, nrow: int = 8) -> None:
        """
        Shows the images and their captions if a sufficient number of them has been accumulated. If the images and
        captions are shown, they are removed from the memory of `ImageSet`.

        :param force: If `True`, shows everything anyway. This may be useful to make the last call of `maybe_show`.
        :param nrow: `nrow` passed to `imshow_tensors`.
        """
        if self.images and (force or len(self.images) >= self.max_num):
            imshow_tensors(*sum(self.images, []), captions=sum(self.captions, []), nrow=nrow)
            self.images.clear()
            self.captions.clear()
