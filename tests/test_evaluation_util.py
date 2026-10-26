import unittest

from latentspace.cnn import Trainer
from latentspace.datasets import DatasetInfo
from latentspace.evaluation_util import EvaluationUtil
from latentspace.generative import WGAN
from latentspace.torch_util import set_random_seed
from tests.helpers import number_of_figures, random_mnist_like_dataset, read_log


class ClassifierEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        set_random_seed(0)
        self.ds = random_mnist_like_dataset()
        self.classifiers = [Trainer("mnist", self.ds.get_train_loader, self.ds.get_test_loader)]

    def test_show_some_predictions(self) -> None:
        before = number_of_figures()
        EvaluationUtil.show_some_predictions(self.classifiers, self.ds)
        self.assertEqual(number_of_figures(), before + 1)

    def test_evaluate_accuracy(self) -> None:
        EvaluationUtil.evaluate_accuracy(self.classifiers, self.ds, max_imgs=8)
        log = read_log()
        self.assertIn("Accuracy of classifier 0 on 8 validation images (noise 0.0)", log)
        self.assertIn("Accuracy of classifier 0 on 8 validation images (noise 0.8)", log)

    def test_evaluate_conventional_robustness(self) -> None:
        EvaluationUtil.evaluate_conventional_robustness(self.classifiers, self.ds, 4, [0.05], [0.1])
        log = read_log()
        self.assertIn("(scaled_l_2) ||dx|| <= 0.050000, accuracy on 4 images", log)
        self.assertIn("(     l_inf) ||dx|| <= 0.100000, accuracy on 4 images", log)

    def test_evaluate_conventional_adversarial_severity(self) -> None:
        EvaluationUtil.evaluate_conventional_adversarial_severity(self.classifiers, self.ds, 4, 0.6, 1.2)
        log = read_log()
        self.assertIn("Adversarial severity of classifier 0 with scaled_l_2 norm", log)
        self.assertIn("Adversarial severity of classifier 0 with      l_inf norm", log)

    def test_generate_images_with_classifier(self) -> None:
        before = number_of_figures()
        EvaluationUtil.generate_images_with_classifier(
            self.classifiers, self.ds, 2, 0.1, "scaled_l_2", 0.5, pairs_in_line=2, estimation_batches=2
        )
        self.assertEqual(number_of_figures(), before + 1)
        # misclassification gradients are restored
        self.assertTrue(self.classifiers[0].misclassification_gradients)


class GenerativeModelEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        set_random_seed(0)
        self.gm = WGAN(DatasetInfo.MNIST, 5, random_mnist_like_dataset())

    def test_show_generated_images(self) -> None:
        before = number_of_figures()
        EvaluationUtil.show_generated_images(self.gm, 2, 3)
        self.assertEqual(number_of_figures(), before + 2)

    def test_show_reconstructed_images(self) -> None:
        before = number_of_figures()
        EvaluationUtil.show_reconstructed_images(self.gm, 1, 1)
        self.assertEqual(number_of_figures(), before + 1)

    def test_show_reconstruction_distance_statistics(self) -> None:
        EvaluationUtil.show_reconstruction_distance_statistics(self.gm, 2)


if __name__ == "__main__":
    unittest.main()
