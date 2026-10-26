import unittest
from collections.abc import Callable, Iterator

import torch

from latentspace.loader_decorators import class_specific_loader, fixed_length_loader, restarting_cycle


def batches(labels: list[int], batch_size: int) -> Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]:
    """Loader-returning function producing batches of items (equal to their indices) with the given labels."""

    def loader() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for i in range(0, len(labels) - batch_size + 1, batch_size):
            yield torch.arange(i, i + batch_size), torch.tensor(labels[i : i + batch_size])

    return loader


class ClassSpecificLoaderTest(unittest.TestCase):
    def test_keeps_only_target_label_and_batch_size(self) -> None:
        labels = [0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1]  # 8 items of label 1
        result = list(class_specific_loader(1, batches(labels, 4))())
        self.assertEqual(len(result), 2)
        for items, batch_labels in result:
            self.assertEqual(len(items), 4)
            self.assertEqual(batch_labels, [1] * 4)
            self.assertTrue(all(labels[i] == 1 for i in items.tolist()))
        # all items of label 1 are produced, in order
        produced = torch.cat([items for items, _ in result]).tolist()
        self.assertEqual(produced, [i for i, label in enumerate(labels) if label == 1])

    def test_incomplete_last_batch_is_dropped(self) -> None:
        labels = [1, 1, 1, 0, 1, 0, 0, 0]  # 4 items of label 1, batch size 4
        self.assertEqual(len(list(class_specific_loader(1, batches(labels, 4))())), 1)
        self.assertEqual(len(list(class_specific_loader(0, batches(labels[:7] + [1], 4))())), 0)


class FixedLengthLoaderTest(unittest.TestCase):
    def test_limits_the_parent_loader(self) -> None:
        loader = fixed_length_loader(8, batches([0] * 12, 4))
        self.assertEqual(sum(len(items) for items, _ in loader()), 8)

    def test_number_of_images_is_rounded_up_to_batch_size(self) -> None:
        loader = fixed_length_loader(10, batches([0] * 12, 4))
        self.assertEqual(sum(len(items) for items, _ in loader()), 12)

    def test_restarts_the_parent_loader(self) -> None:
        loader = fixed_length_loader(20, batches([0] * 12, 4))
        items = torch.cat([items for items, _ in loader()]).tolist()
        self.assertEqual(items, list(range(12)) + list(range(8)))

    def test_without_restarts_stops_at_the_end(self) -> None:
        loader = fixed_length_loader(20, batches([0] * 12, 4), restarts=False)
        self.assertEqual(sum(len(items) for items, _ in loader()), 12)


class RestartingCycleTest(unittest.TestCase):
    def test_cycles_endlessly(self) -> None:
        cycle = restarting_cycle(lambda: [1, 2, 3])
        self.assertEqual([next(cycle) for _ in range(7)], [1, 2, 3, 1, 2, 3, 1])

    def test_calls_the_function_again_on_each_pass(self) -> None:
        calls = []

        def iterable_fn() -> list[int]:
            calls.append(len(calls))
            return [10 * len(calls), 10 * len(calls) + 1]

        cycle = restarting_cycle(iterable_fn)
        self.assertEqual([next(cycle) for _ in range(6)], [10, 11, 20, 21, 30, 31])
        self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
