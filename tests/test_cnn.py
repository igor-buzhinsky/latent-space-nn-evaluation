import os
import tempfile
import unittest

import torch

from latentspace.cnn import ColoredNet128, EarlyStoppingMonitor, MNISTNet, Trainer, Unit
from latentspace.paths import REPO_ROOT
from latentspace.torch_util import set_random_seed, to_device
from tests.helpers import random_mnist_like_dataset


def mnist_trainer(unit_type: int = 0) -> Trainer:
    ds = random_mnist_like_dataset()
    return Trainer("mnist", ds.get_train_loader, ds.get_test_loader, unit_type)


def mnist_images(num_images: int) -> torch.Tensor:
    return to_device(torch.rand(num_images, 1, 28, 28) * 2 - 1)


class NetworkTest(unittest.TestCase):
    def test_mnist_net_output_shape(self) -> None:
        for unit_type in [0, 1, 2]:
            with self.subTest(unit_type=unit_type):
                self.assertEqual(MNISTNet(unit_type)(torch.zeros(3, 1, 28, 28)).shape, (3, 10))

    def test_colored_net_output_shape(self) -> None:
        for unit_type in [0, 1, 2]:
            with self.subTest(unit_type=unit_type):
                self.assertEqual(ColoredNet128(2, unit_type)(torch.zeros(2, 3, 128, 128)).shape, (2, 2))

    def test_unit_halves_spatial_size(self) -> None:
        self.assertEqual(Unit(3, 8, 0.1)(torch.zeros(1, 3, 16, 16)).shape, (1, 8, 8, 8))

    def test_unknown_unit_type_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Unit(3, 8, 0.1, unit_type=3)


class TrainerTest(unittest.TestCase):
    def setUp(self) -> None:
        set_random_seed(0)
        self.trainer = mnist_trainer()

    def test_unknown_dataset_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            Trainer("cifar", lambda: [], lambda: [])

    def test_predict(self) -> None:
        predictions = self.trainer.predict(mnist_images(5))
        self.assertEqual(predictions.shape, (5,))
        self.assertTrue(((predictions >= 0) & (predictions < 10)).all())

    def test_predict_probs_sum_to_one(self) -> None:
        probs = self.trainer.predict_probs(mnist_images(4))
        self.assertEqual(probs.shape, (4, 10))
        torch.testing.assert_close(probs.sum(dim=1), to_device(torch.ones(4)))

    def test_predict_with_gradient_computes_input_gradient(self) -> None:
        x = mnist_images(2).requires_grad_(True)
        predictions, losses = self.trainer.predict_with_gradient(x, [3, 7])
        self.assertEqual(predictions.shape, (2,))
        self.assertEqual(losses.shape, (2,))
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        # the misclassification "loss" is a difference of probabilities
        self.assertTrue(((losses >= -1) & (losses <= 1)).all())

    def test_save_and_restore_params_from_disk(self) -> None:
        x = mnist_images(4)
        expected = self.trainer.predict_probs(x)
        with tempfile.TemporaryDirectory() as dirname:
            filename = os.path.join(dirname, "classifier.bin")
            self.trainer.save_params_to_disk(filename)
            other = mnist_trainer()
            other.restore_params_from_disk(filename)
        torch.testing.assert_close(other.predict_probs(x), expected)

    def test_random_init_changes_params(self) -> None:
        before = [p.clone() for p in self.trainer.model.parameters()]
        self.trainer.random_init()
        after = list(self.trainer.model.parameters())
        self.assertTrue(any(not torch.equal(b, a) for b, a in zip(before, after, strict=True)))

    def test_accuracy_counts_full_batches(self) -> None:
        ds = random_mnist_like_dataset(num_images=10)
        accuracy, total = self.trainer.accuracy(lambda: ds.get_test_loader(batch_size=4))
        self.assertEqual(total, 8)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

    def test_accuracy_with_noise_multiplier(self) -> None:
        ds = random_mnist_like_dataset(num_images=8)
        _, total = self.trainer.accuracy(lambda: ds.get_test_loader(batch_size=4), 0.5, 3)
        self.assertEqual(total, 24)

    def test_fit_saves_parameters(self) -> None:
        previous_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as dirname:
            os.chdir(dirname)  # fit writes TensorBoard logs to the current directory
            try:
                backup = os.path.join(dirname, "backup.bin")
                self.trainer.fit([0.1] * 10, epochs=1, lr=1e-3, disk_backup_filename=backup)
                self.assertTrue(os.path.isfile(backup))
            finally:
                os.chdir(previous_dir)


class EarlyStoppingMonitorTest(unittest.TestCase):
    def test_stops_and_restores_params_when_metric_decreases(self) -> None:
        trainer = mnist_trainer()
        monitor = EarlyStoppingMonitor()
        x = mnist_images(4)
        with tempfile.TemporaryDirectory() as dirname:
            backup = os.path.join(dirname, "backup.bin")
            monitor.update_metric(trainer, 0.5, 0, backup)
            self.assertFalse(monitor.should_stop)
            self.assertTrue(os.path.isfile(backup))
            best = trainer.predict_probs(x)
            trainer.random_init()
            monitor.update_metric(trainer, 0.4, 1, backup)
        self.assertTrue(monitor.should_stop)
        torch.testing.assert_close(trainer.predict_probs(x), best)


class BundledClassifiersTest(unittest.TestCase):
    def test_bundled_mnist_classifiers_can_be_loaded(self) -> None:
        for unit_type in [0, 1, 2]:
            for prefix in ["oneepoch", "plain", "conventional", "robust", "both"]:
                with self.subTest(unit_type=unit_type, prefix=prefix):
                    trainer = mnist_trainer(unit_type)
                    trainer.restore_params_from_disk(
                        str(REPO_ROOT / f"classifiers_architecture{unit_type}" / f"{prefix}_MNIST.bin")
                    )


if __name__ == "__main__":
    unittest.main()
