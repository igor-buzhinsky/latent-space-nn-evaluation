import os
import tempfile
import unittest

import torch

from latentspace.gan import GAN, SpectralNormConv2d, SpectralNormLinear
from latentspace.torch_util import set_random_seed, to_device
from tests.helpers import random_mnist_like_dataset


def spectral_norm_of(weight: torch.Tensor | torch.nn.Module) -> float:
    # the normalized weight is set by a forward pre-hook, so its static type is ambiguous
    assert isinstance(weight, torch.Tensor)
    return torch.linalg.matrix_norm(weight.reshape(weight.shape[0], -1), ord=2).item()


class SpectralNormTest(unittest.TestCase):
    def test_linear_weight_is_normalized_after_power_iterations(self) -> None:
        set_random_seed(0)
        layer = SpectralNormLinear(20, 10)
        for _ in range(30):
            layer(torch.randn(1, 20))
        self.assertAlmostEqual(spectral_norm_of(layer.linear.weight), 1.0, places=3)

    def test_conv_weight_is_normalized_after_power_iterations(self) -> None:
        set_random_seed(0)
        layer = SpectralNormConv2d(3, 4, kernel_size=3, padding=1)
        for _ in range(30):
            output = layer(torch.randn(1, 3, 8, 8))
        self.assertEqual(output.shape, (1, 4, 8, 8))
        self.assertAlmostEqual(spectral_norm_of(layer.conv.weight), 1.0, places=3)


class GANTest(unittest.TestCase):
    def setUp(self) -> None:
        set_random_seed(0)
        self.gan = GAN()

    def test_generated_images(self) -> None:
        images = self.gan.generate(5)
        self.assertEqual(images.shape, (5, 1, 28, 28))
        self.assertTrue(((images >= -1) & (images <= 1)).all())

    def test_decode_is_deterministic(self) -> None:
        latent = to_device(torch.randn(2, self.gan.latent_dim))
        torch.testing.assert_close(self.gan.decode(latent), self.gan.decode(latent))

    def test_discriminator_output_shape(self) -> None:
        self.assertEqual(self.gan.discriminator(self.gan.generate(3)).shape, (3, 1))

    def test_save_and_restore_params_from_disk(self) -> None:
        latent = to_device(torch.randn(2, self.gan.latent_dim))
        expected = self.gan.decode(latent)
        with tempfile.TemporaryDirectory() as dirname:
            filename = os.path.join(dirname, "gan.bin")
            self.gan.save_params_to_disk(filename)
            self.assertTrue(os.path.isfile(filename + ".generator"))
            self.assertTrue(os.path.isfile(filename + ".discriminator"))
            other = GAN()
            other.restore_params_from_disk(filename)
        torch.testing.assert_close(other.decode(latent), expected)

    def test_restore_params_without_saved_params_fails(self) -> None:
        with self.assertRaises(RuntimeError):
            self.gan.restore_params()

    def test_bundled_weights_can_be_loaded(self) -> None:
        for label in range(10):
            with self.subTest(label=label):
                self.gan.restore_params_from_disk(f"mnist-gan/dumped_weights_{label}.bin")

    def test_warm_spectral_norms_runs_over_loader(self) -> None:
        ds = random_mnist_like_dataset(num_images=8)
        self.gan.warm_spectral_norms(lambda: ds.get_test_loader(batch_size=4))

    def test_fit_saves_parameters(self) -> None:
        ds = random_mnist_like_dataset(num_images=40)
        previous_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as dirname:
            os.chdir(dirname)  # fit writes TensorBoard logs to the current directory
            try:
                backup = os.path.join(dirname, "gan.bin")
                self.gan.fit(
                    lambda: ds.get_train_loader(batch_size=4), epochs=1, n_critic=2, disk_backup_filename=backup
                )
                self.assertTrue(os.path.isfile(backup + ".generator"))
            finally:
                os.chdir(previous_dir)


if __name__ == "__main__":
    unittest.main()
