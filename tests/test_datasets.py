import unittest
from collections.abc import Iterator

import torch

from latentspace.datasets import DatasetInfo, DatasetWrapper, merge_loaders
from tests.helpers import TensorDatasetWrapper


class DatasetInfoTest(unittest.TestCase):
    def test_supported_datasets(self) -> None:
        self.assertEqual([d.name for d in DatasetInfo], ["MNIST", "CelebA128Gender", "LSUN128", "ImageNet"])
        self.assertIs(DatasetInfo["MNIST"], DatasetInfo.MNIST)


class MergeLoadersTest(unittest.TestCase):
    @staticmethod
    def constant_loader(label: int, num_batches: int):
        def loader(batch_size: int, shuffle: bool) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            for _ in range(num_batches):
                yield torch.full((batch_size, 1), float(label)), torch.full((batch_size,), label)

        return loader

    def test_batches_alternate_between_loaders_without_shuffling(self) -> None:
        loaders = [self.constant_loader(0, 3), self.constant_loader(1, 3)]
        batches = list(merge_loaders(loaders, batch_size=4, shuffle=False))
        self.assertEqual(len(batches), 6)
        self.assertEqual([batch_labels.tolist() for _, batch_labels in batches[:2]], [[0] * 4, [1] * 4])

    def test_shuffling_preserves_items_and_batch_size(self) -> None:
        loaders = [self.constant_loader(0, 2), self.constant_loader(1, 2)]
        batches = list(merge_loaders(loaders, batch_size=4, shuffle=True))
        self.assertTrue(all(len(items) == 4 for items, _ in batches))
        all_labels = torch.cat([batch_labels for _, batch_labels in batches])
        self.assertEqual(all_labels.tolist().count(0), 8)
        self.assertEqual(all_labels.tolist().count(1), 8)
        for items, batch_labels in batches:
            torch.testing.assert_close(items[:, 0], batch_labels.float())

    def test_stops_when_shortest_loader_ends(self) -> None:
        loaders = [self.constant_loader(0, 1), self.constant_loader(1, 5)]
        self.assertEqual(len(list(merge_loaders(loaders, batch_size=2, shuffle=False))), 2)


class DatasetWrapperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = torch.arange(10) % 3
        self.ds = TensorDatasetWrapper(torch.randn(10, 1, 8, 8), self.labels, ["a", "b", "c"])

    def test_loaders_drop_incomplete_batches(self) -> None:
        batches = list(self.ds.get_test_loader(batch_size=4, shuffle=False))
        self.assertEqual([len(images) for images, _ in batches], [4, 4])
        self.assertEqual(len(list(self.ds.get_train_loader(batch_size=3))), 3)
        self.assertEqual(len(list(self.ds.get_unaugmented_train_loader(batch_size=5))), 2)

    def test_default_batch_sizes(self) -> None:
        images, _ = next(iter(self.ds.get_test_loader()))
        self.assertEqual(len(images), DatasetWrapper.test_batch_size)

    def test_labels_are_preserved_without_shuffling(self) -> None:
        labels = torch.cat([batch_labels.cpu() for _, batch_labels in self.ds.get_test_loader(5, shuffle=False)])
        torch.testing.assert_close(labels, self.labels)

    def test_unique_label_overrides_labels(self) -> None:
        self.ds.unique_label = 2
        _, labels = next(iter(self.ds.get_test_loader(4)))
        self.assertEqual(labels.tolist(), [2] * 4)

    def test_class_names(self) -> None:
        self.assertEqual(self.ds.prediction_indices_to_classes(torch.tensor([2, 0])), ["c", "a"])
        self.assertEqual(self.ds.prediction_indices_to_printed_classes([1]), ["b"])

    def test_missing_dataset_part_is_reported(self) -> None:
        self.ds.testset = None
        with self.assertRaises(AssertionError):
            self.ds.get_test_loader()


if __name__ == "__main__":
    unittest.main()
