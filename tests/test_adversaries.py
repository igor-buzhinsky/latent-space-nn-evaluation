import unittest
from collections.abc import Callable

import numpy as np
import torch

from latentspace.adversaries import NopAdversary, PGDAdversary
from latentspace.torch_util import set_random_seed, to_device

DIM = 16


def linear_objective(threshold: float) -> tuple[torch.Tensor, Callable[[torch.Tensor], tuple[torch.Tensor, float]]]:
    """
    Objective `v · w - threshold` with a fixed unit vector `w`. A perturbation is adversarial if the objective is
    positive, i.e., the minimum adversarial perturbation has L2 norm `threshold` (scaled L2 norm `threshold / 4`).
    """
    w = to_device(torch.ones(1, DIM) / np.sqrt(DIM))

    def get_gradient(v: torch.Tensor) -> tuple[torch.Tensor, float]:
        return w, (v * w).sum().item() - threshold

    return w, get_gradient


def scaled_norm(x: torch.Tensor) -> float:
    return x.norm().item() / np.sqrt(x.numel())


class NopAdversaryTest(unittest.TestCase):
    def test_returns_initial_vector(self) -> None:
        x = torch.randn(1, DIM)
        self.assertIs(NopAdversary().perturb(x, linear_objective(1.0)[1]), x)


class PGDAdversaryTest(unittest.TestCase):
    def setUp(self) -> None:
        set_random_seed(0)
        self.origin = to_device(torch.zeros(1, DIM))

    def test_finds_adversarial_perturbation_within_bound(self) -> None:
        _, get_gradient = linear_objective(1.0)
        adversary = PGDAdversary(rho=1.0, steps=50, step_size=0.1, random_start=False, verbose=0)
        perturbed = adversary.perturb(self.origin, get_gradient)
        self.assertGreater(get_gradient(perturbed)[1], 0)
        self.assertLessEqual(scaled_norm(perturbed), 1.0 + 1e-6)

    def test_stops_as_soon_as_perturbation_is_adversarial(self) -> None:
        _, get_gradient = linear_objective(1.0)
        adversary = PGDAdversary(rho=1.0, steps=50, step_size=0.1, random_start=False, verbose=0)
        perturbed = adversary.perturb(self.origin, get_gradient)
        # each step increases the scaled norm by rho * step_size, and 3 steps are needed to exceed the threshold
        self.assertAlmostEqual(scaled_norm(perturbed), 0.3, places=5)

    def test_already_adversarial_vector_is_not_changed(self) -> None:
        _, get_gradient = linear_objective(-1.0)
        adversary = PGDAdversary(rho=1.0, steps=50, random_start=False, verbose=0)
        torch.testing.assert_close(adversary.perturb(self.origin, get_gradient), self.origin)

    def test_unreachable_objective_leads_to_perturbation_on_the_bound(self) -> None:
        _, get_gradient = linear_objective(1e6)
        adversary = PGDAdversary(rho=0.5, steps=30, step_size=0.1, random_start=True, verbose=0)
        perturbed = adversary.perturb(self.origin, get_gradient)
        self.assertAlmostEqual(scaled_norm(perturbed), 0.5, places=5)

    def test_l_inf_perturbation_is_clamped(self) -> None:
        _, get_gradient = linear_objective(1e6)
        adversary = PGDAdversary(rho=0.2, steps=30, step_size=0.1, verbose=0, norm="l_inf")
        perturbed = adversary.perturb(self.origin, get_gradient)
        torch.testing.assert_close(perturbed, torch.full_like(perturbed, 0.2))

    def test_min_repeat_mode_shrinks_the_perturbation(self) -> None:
        _, get_gradient = linear_objective(1.0)
        adversary = PGDAdversary(rho=2.0, steps=50, step_size=0.05, verbose=0, n_repeat=10, repeat_mode="min")
        perturbed = adversary.perturb(self.origin, get_gradient)
        self.assertGreater(get_gradient(perturbed)[1], 0)
        # the minimum adversarial perturbation has scaled norm 0.25
        self.assertLess(scaled_norm(perturbed), 0.35)

    def test_unit_sphere_normalization(self) -> None:
        _, get_gradient = linear_objective(1e6)
        x = to_device(torch.randn(1, DIM))
        adversary = PGDAdversary(rho=0.5, steps=10, verbose=0, unit_sphere_normalization=True)
        self.assertAlmostEqual(scaled_norm(adversary.perturb(x, get_gradient)), 1.0, places=5)

    def test_invalid_arguments_are_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            PGDAdversary(norm="l_1")
        with self.assertRaises(AssertionError):
            PGDAdversary(n_repeat=3)
        with self.assertRaises(AssertionError):
            PGDAdversary(norm="l_inf", unit_sphere_normalization=True)


if __name__ == "__main__":
    unittest.main()
