import unittest

import torch

from latentspace.cnn import Trainer
from latentspace.datasets import DatasetInfo
from latentspace.generative import WGAN
from latentspace.paths import REPO_ROOT
from latentspace.torch_util import set_random_seed, to_device
from tests.helpers import random_mnist_like_dataset


class WGANTest(unittest.TestCase):
    def setUp(self) -> None:
        set_random_seed(0)
        self.ds = random_mnist_like_dataset()
        self.gm = WGAN(DatasetInfo.MNIST, 3, self.ds)

    def test_properties(self) -> None:
        self.assertEqual(self.gm.latent_dim, 64)
        self.assertEqual(self.gm.resolution, 28)
        self.assertEqual(self.gm.unique_label, 3)
        self.assertEqual(self.gm.subclass, "3")

    def test_generate(self) -> None:
        images = self.gm.generate(4, detach=True)
        self.assertEqual(images.shape, (4, 1, 28, 28))
        self.assertFalse(images.requires_grad)

    def test_decode_with_and_without_detach(self) -> None:
        latent = to_device(torch.randn(2, 64, requires_grad=True))
        self.assertFalse(self.gm.decode(latent).requires_grad)
        self.assertTrue(self.gm.decode(latent, detach=False).requires_grad)

    def test_encode_reconstructs_generated_image(self) -> None:
        image = self.gm.generate(1, detach=True)
        latent = self.gm.encode(image)
        self.assertEqual(latent.shape, (1, 64))
        reconstruction_error = (self.gm.decode(latent) - image).abs().mean().item()
        random_error = (self.gm.generate(1, detach=True) - image).abs().mean().item()
        self.assertLess(reconstruction_error, random_error)

    def test_sampler_yields_images_of_the_class(self) -> None:
        images, labels = next(self.gm.get_sampler(batch_size=2))
        self.assertEqual(images.shape, (2, 1, 28, 28))
        self.assertEqual(labels, [3, 3])

    def test_destroy(self) -> None:
        self.gm.destroy()
        self.assertFalse(hasattr(self.gm, "g"))

    def test_only_mnist_is_supported(self) -> None:
        with self.assertRaises(ValueError):
            WGAN(DatasetInfo.LSUN128, 0, self.ds)


class BundledModelsTest(unittest.TestCase):
    def test_robust_classifier_recognizes_generated_digits(self) -> None:
        set_random_seed(0)
        ds = random_mnist_like_dataset()
        classifier = Trainer("mnist", ds.get_train_loader, ds.get_test_loader, unit_type=0)
        classifier.restore_params_from_disk(str(REPO_ROOT / "classifiers_architecture0" / "robust_MNIST.bin"))
        for label in range(10):
            with self.subTest(label=label):
                images = WGAN(DatasetInfo.MNIST, label, ds).generate(20, detach=True)
                accuracy = (classifier.predict(images) == label).float().mean().item()
                self.assertGreaterEqual(accuracy, 0.8)


if __name__ == "__main__":
    unittest.main()
