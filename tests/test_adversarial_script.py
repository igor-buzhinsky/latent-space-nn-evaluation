import unittest

import numpy as np

from Adversarial import build_parser, images_per_class, min_total_images

MNIST_PROPORTIONS = np.repeat(0.1, 10)
CELEBA_PROPORTIONS = np.array([0.583, 0.417])


class ImagesPerClassTest(unittest.TestCase):
    def test_images_are_split_in_proportion_to_class_frequencies(self) -> None:
        np.testing.assert_array_equal(images_per_class(CELEBA_PROPORTIONS, 100), [58, 42])
        np.testing.assert_array_equal(images_per_class(MNIST_PROPORTIONS, 30), np.repeat(3, 10))

    def test_small_totals_leave_classes_without_images(self) -> None:
        self.assertEqual(images_per_class(MNIST_PROPORTIONS, 3).sum(), 0)


class MinTotalImagesTest(unittest.TestCase):
    def test_every_class_gets_an_image_at_the_minimum_but_not_below(self) -> None:
        for proportions in [MNIST_PROPORTIONS, CELEBA_PROPORTIONS, np.array([0.5, 0.5])]:
            with self.subTest(proportions=proportions):
                total = min_total_images(proportions)
                self.assertGreaterEqual(images_per_class(proportions, total).min(), 1)
                self.assertEqual(images_per_class(proportions, total - 1).min(), 0)


class BuildParserTest(unittest.TestCase):
    def test_parses_generation_command(self) -> None:
        args = build_parser().parse_args(
            ["--dataset", "MNIST", "--command", "generate_minimum", "--num_images", "10", "--noise_epsilon", "0.5"]
            + ["--classifier_filenames", "classifiers_architecture0/plain_MNIST.bin"]
        )
        self.assertEqual(args.num_images, 10)


if __name__ == "__main__":
    unittest.main()
