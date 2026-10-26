import unittest

import numpy as np
import torch

from latentspace.plotting import ImageSet, get_kde_bandwidth, imshow, imshow_tensors
from tests import helpers


def number_of_figures() -> int:
    return helpers.number_of_figures("imshow_")


class ImshowTest(unittest.TestCase):
    def test_imshow_saves_figure(self) -> None:
        before = number_of_figures()
        imshow(torch.rand(3, 16, 16) * 2 - 1)
        self.assertEqual(number_of_figures(), before + 1)

    def test_imshow_tensors_with_grayscale_images_and_captions(self) -> None:
        before = number_of_figures()
        imshow_tensors(torch.rand(2, 1, 28, 28) * 2 - 1, torch.rand(1, 1, 28, 28), captions=["a", "b\nc", "d"])
        self.assertEqual(number_of_figures(), before + 1)

    def test_imshow_tensors_rejects_non_string_captions(self) -> None:
        with self.assertRaises(AssertionError):
            imshow_tensors(torch.rand(1, 3, 8, 8), captions=[1])  # pyrefly: ignore[bad-argument-type]


class ImageSetTest(unittest.TestCase):
    def test_images_are_shown_when_enough_are_accumulated(self) -> None:
        image_set = ImageSet(max_num=2)
        before = number_of_figures()
        image_set.append([torch.rand(1, 3, 8, 8)], ["first"])
        image_set.maybe_show()
        self.assertEqual(number_of_figures(), before)
        image_set.append([torch.rand(1, 3, 8, 8)])
        image_set.maybe_show()
        self.assertEqual(number_of_figures(), before + 1)
        self.assertEqual(image_set.images, [])

    def test_force_shows_remaining_images(self) -> None:
        image_set = ImageSet(max_num=10)
        before = number_of_figures()
        image_set.append([torch.rand(1, 3, 8, 8)])
        image_set.maybe_show(force=True)
        self.assertEqual(number_of_figures(), before + 1)


class GetKdeBandwidthTest(unittest.TestCase):
    def test_bandwidth_scales_with_data(self) -> None:
        x = np.random.default_rng(0).normal(size=1000)
        bandwidth = get_kde_bandwidth(x)
        self.assertGreater(bandwidth, 0)
        self.assertAlmostEqual(get_kde_bandwidth(10 * x), 10 * bandwidth)


if __name__ == "__main__":
    unittest.main()
