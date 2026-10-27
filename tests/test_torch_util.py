import random
import unittest

import numpy as np
import torch

from latentspace.torch_util import (
    DEVICE,
    init_weights,
    normalize_latent,
    number_of_trainable_parameters,
    optimizable_clone,
    set_param_requires_grad,
    set_random_seed,
    to_device,
    to_numpy,
)


class ToDeviceTest(unittest.TestCase):
    def test_tensor_is_moved_to_device(self) -> None:
        x = to_device(torch.zeros(3))
        self.assertEqual(x.device.type, torch.device(DEVICE).type)

    def test_module_is_moved_to_device(self) -> None:
        module = to_device(torch.nn.Linear(2, 2))
        self.assertEqual(next(module.parameters()).device.type, torch.device(DEVICE).type)


class ToNumpyTest(unittest.TestCase):
    def test_converts_tensor_requiring_gradients(self) -> None:
        x = to_device(torch.tensor([1.0, 2.0, 3.0], requires_grad=True))
        array = to_numpy(x)
        self.assertIsInstance(array, np.ndarray)
        np.testing.assert_array_equal(array, [1.0, 2.0, 3.0])


class OptimizableCloneTest(unittest.TestCase):
    def test_clone_is_a_leaf_with_gradients(self) -> None:
        x = torch.ones(2, 3)
        clone = optimizable_clone(x)
        self.assertTrue(clone.requires_grad)
        self.assertTrue(clone.is_leaf)
        torch.testing.assert_close(clone.detach().cpu(), x)

    def test_clone_does_not_share_memory(self) -> None:
        x = to_device(torch.ones(3))
        clone = optimizable_clone(x)
        with torch.no_grad():
            clone += 1
        torch.testing.assert_close(x, to_device(torch.ones(3)))


class InitWeightsTest(unittest.TestCase):
    def test_conv_and_linear_layers_are_reinitialized(self) -> None:
        conv, linear = torch.nn.Conv2d(1, 2, 3), torch.nn.Linear(4, 5)
        old_conv_weight, old_linear_weight = conv.weight.clone(), linear.weight.clone()
        torch.nn.Sequential(conv, linear).apply(init_weights)
        self.assertFalse(torch.equal(conv.weight, old_conv_weight))
        self.assertFalse(torch.equal(linear.weight, old_linear_weight))
        torch.testing.assert_close(linear.bias, torch.zeros(5))

    def test_other_layers_are_not_changed(self) -> None:
        batch_norm = torch.nn.BatchNorm2d(3)
        batch_norm.apply(init_weights)
        torch.testing.assert_close(batch_norm.weight, torch.ones(3))


class SetParamRequiresGradTest(unittest.TestCase):
    def test_sets_flag_for_all_parameters(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        set_param_requires_grad(model, False)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        set_param_requires_grad(model, True)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))


class NumberOfTrainableParametersTest(unittest.TestCase):
    def test_counts_only_trainable_parameters(self) -> None:
        linear = torch.nn.Linear(3, 2)
        self.assertEqual(number_of_trainable_parameters(linear), 3 * 2 + 2)
        linear.bias.requires_grad_(False)
        self.assertEqual(number_of_trainable_parameters(linear), 3 * 2)


class SetRandomSeedTest(unittest.TestCase):
    def test_same_seed_gives_same_random_numbers(self) -> None:
        def draw() -> tuple[float, float, list[float]]:
            return random.random(), float(np.random.rand()), torch.rand(3).tolist()

        set_random_seed(123)
        first = draw()
        set_random_seed(123)
        self.assertEqual(draw(), first)

    def test_negative_seed_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            set_random_seed(-1)


class NormalizeLatentTest(unittest.TestCase):
    def test_each_vector_gets_unit_scaled_norm(self) -> None:
        x = torch.randn(4, 16) * torch.tensor([[0.5], [1.0], [2.0], [10.0]])
        normalized = normalize_latent(x)
        scaled_norms = normalized.norm(dim=1) / np.sqrt(16)
        torch.testing.assert_close(scaled_norms, torch.ones(4))

    def test_directions_are_preserved(self) -> None:
        x = torch.randn(3, 8)
        cosine = torch.nn.functional.cosine_similarity(normalize_latent(x), x, dim=1)
        torch.testing.assert_close(cosine, torch.ones(3))


if __name__ == "__main__":
    unittest.main()
