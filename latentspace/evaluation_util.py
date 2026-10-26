import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch

from latentspace.adversarial_generation import get_conventional_perturb
from latentspace.adversaries import PGDAdversary
from latentspace.cnn import Trainer
from latentspace.datasets import DatasetWrapper
from latentspace.generative import GenerativeModel
from latentspace.loader_decorators import fixed_length_loader
from latentspace.logging_util import LogUtil
from latentspace.plotting import ImageSet, imshow_tensors
from latentspace.torch_util import to_device


class EvaluationUtil:
    @staticmethod
    def show_some_predictions(classifiers: list[Trainer], ds: DatasetWrapper) -> None:
        """
        Shows several images, and prints/logs several predictions of the supplied classifiers on them.

        :param classifiers: List of classifiers.
        :param ds: `DatasetWrapper` from which test images will be taken.
        """
        images, labels = next(iter(ds.get_test_loader()))
        imshow_tensors(images, captions=ds.prediction_indices_to_printed_classes(labels))
        for i, c in enumerate(classifiers):
            LogUtil.info(f"Predicted {i}: {ds.prediction_indices_to_classes(c.predict(images))}")

    @staticmethod
    def evaluate_accuracy(
        classifiers: list[Trainer],
        ds: DatasetWrapper,
        max_imgs: int | None = None,
        noise_evaluation_multiplier: int = 1,
    ) -> None:
        """
        Evaluates the conventional accuracy of the supplied classifiers.
        This is done for clean images and for images corrupted with Gaussian noise N(0, 0.8^2 I).

        :param classifiers: List of classifiers.
        :param ds: `DatasetWrapper` from which test images will be taken.
        :param max_imgs: Limits the number of images used in evaluation (the actual number of images will be a multiple
            of batch size). If `None`, evaluation will be done for the full number of images produced by the test
            loader.
        :param noise_evaluation_multiplier: Evaluation on noise-corrupted images will be done
            `noise_evaluation_multiplier` times.
        """
        loader = ds.get_test_loader
        if max_imgs is not None:
            loader = fixed_length_loader(max_imgs, loader, False)
        for i, c in enumerate(classifiers):
            for noise_sigma in [0, 0.8]:
                accuracy, total = c.accuracy(loader, noise_sigma, noise_evaluation_multiplier)
                acc_str = f"{accuracy * 100:.2f}"
                LogUtil.info(
                    f"Accuracy of classifier {i} on {total} validation images (noise {noise_sigma:.1f}): {acc_str:>6}%"
                )

    @staticmethod
    def evaluate_conventional_robustness(
        classifiers: list[Trainer],
        ds: DatasetWrapper,
        max_imgs: int,
        l_2_bounds: list[float],
        l_inf_bounds: list[float],
    ) -> None:
        """
        Evaluates the conventional robustness of the supplied classifiers as accuracy on adversarial perturbations.
        Uses L2 and L-inf norms.

        :param classifiers: List of classifiers.
        :param ds: `DatasetWrapper` from which test images will be taken.
        :param max_imgs: Limits the number of images used in evaluation (the actual number of images will be a multiple
            of batch size).
        :param l_2_bounds: Perturbation L2 norm bounds for which to perform accuracy evaluation.
        :param l_inf_bounds: Perturbation L-inf norm bounds for which to perform accuracy evaluation.
        """
        loader = fixed_length_loader(max_imgs, ds.get_test_loader, False)
        for norm, bounds in [("scaled_l_2", l_2_bounds), ("l_inf", l_inf_bounds)]:
            for bound in bounds:
                adversary = PGDAdversary(bound, 25, 0.1, True, 0, verbose=0, norm=norm)
                for i, c in enumerate(classifiers):
                    perturb = get_conventional_perturb(c, adversary)
                    accuracy, total = c.measure_robustness(perturb, loader, ds, False)
                    acc_str = f"{accuracy * 100:.2f}"
                    LogUtil.info(
                        f"For classifier {i}, ({norm:>10}) ||dx|| <= {bound:.6f}, "
                        f"accuracy on {total} images = {acc_str:>6}%"
                    )

    @staticmethod
    def evaluate_conventional_adversarial_severity(
        classifiers: list[Trainer], ds: DatasetWrapper, max_imgs: int, l_2_bound: float, l_inf_bound: float
    ) -> None:
        """
        Evaluates the conventional adversarial severity (mean norm of minimum adversarial perturbations)
        of the supplied classifiers. Uses L2 and L-inf norms.

        :param classifiers: List of classifiers.
        :param ds: `DatasetWrapper` from which test images will be taken.
        :param max_imgs: Limits the number of images used in evaluation (the actual number of images will be a multiple
            of batch size).
        :param l_2_bound: Maximum possible L2 norm of an adversarial perturbation (should be set to a number somewhat
            greater than the norms of typical perturbations).
        :param l_inf_bound: Maximum possible L-inf norm of an adversarial perturbation (should be set to a number
            somewhat greater than the norms of typical perturbations).
        """
        loader = fixed_length_loader(max_imgs, ds.get_test_loader, False)
        for norm, norm_fn, bound in [
            ("scaled_l_2", lambda x: x.norm() / np.sqrt(x.numel()), l_2_bound),
            ("l_inf", lambda x: x.abs().max(), l_inf_bound),
        ]:
            adversary = PGDAdversary(bound, 50, 0.05, True, 0, verbose=0, n_repeat=15, repeat_mode="min", norm=norm)
            for i, c in enumerate(classifiers):
                perturb = get_conventional_perturb(c, adversary)
                severity, std, total = c.measure_adversarial_severity(perturb, loader, ds, norm_fn, False)
                LogUtil.info(
                    f"Adversarial severity of classifier {i} with {norm:>10} norm = {severity:.8f} "
                    f"(std = {std:.8f}, #images = {total})"
                )

    @staticmethod
    def generate_images_with_classifier(
        classifiers: list[Trainer],
        ds: DatasetWrapper,
        no_classes: int,
        noise_sigma: float,
        norm: str,
        norm_bound: float,
        pairs_in_line: int = 3,
        estimation_batches: int = 100,
        nrow: int = 8,
    ) -> None:
        """
        Uses a classifier to generate images of each class. Starts from a mean of a number of actual images
        with added Gaussian noise. For robust classifiers, this procedure should result in images resembling
        correct class instances.

        :param classifiers: List of classifiers.
        :param ds: `DatasetWrapper` from which test images will be taken.
        :param no_classes: The number of classes in the dataset.
        :param noise_sigma: Noise of magnitude N(0, noise_sigma^2 I) will be added to initial images.
        :param norm: A string specifying a norm as accepted by `PGDAdversary`.
        :param norm_bound: Norm bound for `PGDAdversary`.
        :param pairs_in_line: Pairs of images (original - generated) to present in each row.
        :param estimation_batches: Number of batches to use to compute the initial mean image.
        :param nrow: `nrow` to be passed to `ImageSet`.
        """
        mean_image = torch.stack(
            [batch.mean(dim=0) for batch, _ in itertools.islice(ds.get_test_loader(), 0, estimation_batches)]
        ).mean(dim=0)
        image_optimizer = PGDAdversary(norm_bound, 50, 0.05, False, np.inf, verbose=0, norm=norm)
        for i, classifier in enumerate(classifiers):
            LogUtil.info(f"classifier {i}:")
            classifier.set_misclassification_gradients(False)
            s = ImageSet(pairs_in_line)
            for target_label in range(no_classes):
                image = mean_image + to_device(torch.randn(*mean_image.shape)) * noise_sigma
                perturb = get_conventional_perturb(classifier, image_optimizer)
                s.append([image.unsqueeze(0), perturb(image, target_label).unsqueeze(0)])
                s.maybe_show(nrow=nrow)
            s.maybe_show(True, nrow=nrow)
            classifier.set_misclassification_gradients(True)

    @staticmethod
    def show_reconstructed_images(gm: GenerativeModel, lines: int, images_in_line: int) -> None:
        """
        Shows several images reconstructed by the supplied generative model.

        :param gm: Generative model to use.
        :param lines: Lines of images to show.
        :param images_in_line: Number of (original, reconstructed) image pairs in each line.
        """
        sampler = gm.get_sampler()
        for _ in range(lines):
            images = []
            for _ in range(images_in_line):
                img, _ = next(sampler)
                images += [img, gm.decode(gm.encode(img), detach=True)]
            imshow_tensors(*images, nrow=(images_in_line * 2))

    @staticmethod
    def show_reconstruction_distance_statistics(gm: GenerativeModel, no_images: int) -> None:
        """
        Shows statistics on the distance of reconstructed images (based on the supplied generative model)
        from the original ones.

        :param gm: Generative model to use.
        :param no_images: Number of images to construct statistics for.
        """
        sampler = gm.get_sampler()
        norms, distances = np.empty(no_images), np.empty(no_images)

        def norm(x: torch.Tensor) -> torch.Tensor:
            return x.flatten().norm() / np.sqrt(x.numel())

        for i in range(no_images):
            img, _ = next(sampler)
            norms[i] = norm(img)
            distances[i] = norm(img - gm.decode(gm.encode(img)))
        print(
            f"Scaled norms of original images:       mean={norms.mean():.4f}, "
            f"median={np.median(norms):.4f}, std={norms.std():.4f}"
        )
        print(
            f"Scaled norms of reconstruction errors: mean={distances.mean():.4f}, "
            f"median={np.median(distances):.4f}, std={distances.std():.4f}"
        )
        plt.figure(figsize=(10, 3))
        plt.title("Scaled norms of reconstruction errors")
        plt.hist(distances)
        plt.show()
        plt.close()

    @staticmethod
    def show_generated_images(gm: GenerativeModel, lines: int, images_in_line: int) -> None:
        """
        Shows several images generated by the supplied generative model.

        :param gm: Generative model to use.
        :param lines: Lines of images to show.
        :param images_in_line: Number of generated images in each line.
        """
        for _ in range(lines):
            imshow_tensors(gm.generate(images_in_line, detach=True), nrow=images_in_line)
