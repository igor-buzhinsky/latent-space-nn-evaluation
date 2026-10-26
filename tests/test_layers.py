import unittest

import torch

from latentspace.layers import Flatten, Lambda, NopLayer, Resize, Upsample


class LambdaTest(unittest.TestCase):
    def test_applies_function(self) -> None:
        x = torch.randn(2, 3)
        torch.testing.assert_close(Lambda(lambda t: t * 2 + 1)(x), x * 2 + 1)


class NopLayerTest(unittest.TestCase):
    def test_returns_input(self) -> None:
        x = torch.randn(2, 3)
        self.assertIs(NopLayer()(x), x)


class FlattenTest(unittest.TestCase):
    def test_keeps_batch_dimension(self) -> None:
        self.assertEqual(Flatten()(torch.zeros(5, 3, 4, 4)).shape, (5, 48))


class UpsampleTest(unittest.TestCase):
    def test_doubles_spatial_size(self) -> None:
        self.assertEqual(Upsample()(torch.zeros(2, 3, 7, 9)).shape, (2, 3, 14, 18))

    def test_constant_image_stays_constant(self) -> None:
        torch.testing.assert_close(Upsample()(torch.full((1, 1, 4, 4), 0.5)), torch.full((1, 1, 8, 8), 0.5))


class ResizeTest(unittest.TestCase):
    def test_resizes_to_square(self) -> None:
        self.assertEqual(Resize(224)(torch.zeros(2, 3, 128, 128)).shape, (2, 3, 224, 224))
        self.assertEqual(Resize(32)(torch.zeros(1, 1, 28, 40)).shape, (1, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()
