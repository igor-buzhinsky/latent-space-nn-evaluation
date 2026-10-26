import unittest

import numpy as np
import torch

from latentspace.adversarial_generation import (
    AdversarialGenerator,
    EpsDTransformer,
    RandomPerturbationStatistician,
    TauRhoTransformer,
    get_conventional_perturb,
    get_get_gradient,
)
from latentspace.adversaries import NopAdversary, PGDAdversary
from latentspace.cnn import Trainer
from latentspace.datasets import DatasetInfo
from latentspace.generative import WGAN
from latentspace.torch_util import set_random_seed, to_device
from tests.helpers import number_of_figures, random_mnist_like_dataset, read_log


def robust_mnist_classifier() -> Trainer:
    ds = random_mnist_like_dataset()
    classifier = Trainer("mnist", ds.get_train_loader, ds.get_test_loader, unit_type=0)
    classifier.restore_params_from_disk("classifiers_architecture0/robust_MNIST.bin")
    return classifier


class EpsDTransformerTest(unittest.TestCase):
    def test_known_value(self) -> None:
        self.assertAlmostEqual(EpsDTransformer().eps_to_d(1.0), 1 - 1 / np.sqrt(2))

    def test_conversions_are_inverse(self) -> None:
        transformer = EpsDTransformer()
        for eps in [0.1, 0.5, 1.0, 3.0]:
            self.assertAlmostEqual(transformer.d_to_eps(transformer.eps_to_d(eps)), eps)


class TauRhoTransformerTest(unittest.TestCase):
    def test_conversions_are_inverse(self) -> None:
        transformer = TauRhoTransformer(n_l=64, eps=1.0)
        for rho in [0.1, 0.5, 2.0]:
            self.assertAlmostEqual(transformer.logtau_to_rho(transformer.rho_to_logtau(rho)), rho)

    def test_likelihood_decreases_with_norm(self) -> None:
        transformer = TauRhoTransformer(n_l=64, eps=1.0)
        self.assertGreater(transformer.rho_to_logtau(0.1), transformer.rho_to_logtau(0.5))


class GradientFunctionsTest(unittest.TestCase):
    def setUp(self) -> None:
        set_random_seed(0)
        self.classifier = robust_mnist_classifier()
        self.image = to_device(torch.rand(1, 28, 28) * 2 - 1)

    def test_get_get_gradient_in_image_space(self) -> None:
        get_gradient = get_get_gradient(
            self.classifier, 3, lambda x: x.view(1, 1, 28, 28), lambda x: x, lambda x: x.view(1, -1)
        )
        gradient, loss = get_gradient(self.image.view(1, -1))
        self.assertEqual(gradient.shape, (1, 28 * 28))
        self.assertIsInstance(loss, float)

    def test_conventional_perturbation_keeps_shape_and_bound(self) -> None:
        adversary = PGDAdversary(0.1, 5, 0.1, False, 0, verbose=0)
        perturbed = get_conventional_perturb(self.classifier, adversary)(self.image, 3)
        self.assertEqual(perturbed.shape, self.image.shape)
        scaled_norm = (perturbed - self.image).norm().item() / np.sqrt(self.image.numel())
        self.assertLessEqual(scaled_norm, 0.1 + 1e-5)


class AdversarialGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        set_random_seed(0)
        self.classifier = robust_mnist_classifier()
        self.gm = WGAN(DatasetInfo.MNIST, 1, random_mnist_like_dataset())

    def test_nop_adversary_does_not_change_predictions(self) -> None:
        generator = AdversarialGenerator(self.gm, [self.classifier], True, 0.3)
        generator.generate(NopAdversary(), 3, produce_images=False)
        self.assertEqual(len(generator.recorded_latent_norms[0]), 3)
        self.assertEqual(generator.recorded_modified_successes, generator.recorded_decayed_successes)
        self.assertEqual(generator.recorded_latent_norms[0], [0, 0, 0])
        generator.print_stats(plot=False)

    def test_latent_perturbations_respect_bound(self) -> None:
        generator = AdversarialGenerator(None, [self.classifier], True, 0.3)
        generator.set_generative_model(self.gm)
        adversary = PGDAdversary(0.5, 10, 0.1, True, 0, verbose=0)
        before = number_of_figures("advgen_stats")
        generator.generate(adversary, 3, show_perturbations=True)
        generator.print_stats(plot=True)
        self.assertTrue(all(0 < norm <= 0.5 + 1e-5 for norm in generator.recorded_latent_norms[0]))
        self.assertEqual(number_of_figures("advgen_stats"), before + 1)

    def test_reconstruction_based_generation(self) -> None:
        generator = AdversarialGenerator(self.gm, [self.classifier], False, 0.3)
        generator.generate(NopAdversary(), 1, produce_images=False)
        self.assertEqual(len(generator.recorded_clean_successes[0]), 1)

    def test_statistics_are_cleared(self) -> None:
        generator = AdversarialGenerator(self.gm, [self.classifier], True, 0.3)
        generator.generate(NopAdversary(), 2, produce_images=False)
        generator.generate(NopAdversary(), 1, clear_stat=False, produce_images=False)
        self.assertEqual(len(generator.recorded_decayed_successes[0]), 3)
        generator.generate(NopAdversary(), 1, produce_images=False)
        self.assertEqual(len(generator.recorded_decayed_successes[0]), 1)


class RandomPerturbationStatisticianTest(unittest.TestCase):
    def test_logs_results_for_each_epsilon(self) -> None:
        set_random_seed(0)
        gm = WGAN(DatasetInfo.MNIST, 2, random_mnist_like_dataset())
        RandomPerturbationStatistician(gm, [robust_mnist_classifier()], 1, 2, True, [0.5, 1.0]).process()
        log = read_log()
        self.assertIn("perturbed(0.50)", log)
        self.assertIn("perturbed(1.00)", log)


if __name__ == "__main__":
    unittest.main()
