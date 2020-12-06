from typing import *
import itertools

from ml_util import *
from datasets import *
from adversarial_generation import *
from generative import *
import cnn


class EvaluationUtil:
    @staticmethod
    def show_some_predictions(classifiers: List[cnn.Trainer], ds: DatasetWrapper):
        """
        Shows several images, and prints/logs several predictions of the supplied classifiers on them.
        :param classifiers: list of classifiers.
        :param ds: DatasetWrapper from which test images will be taken.
        """
        images, labels = next(iter(ds.get_test_loader()))
        Util.imshow_tensors(images, captions=ds.prediction_indices_to_printed_classes(labels))
        for i, c in enumerate(classifiers):
            LogUtil.info(f"Predicted {i}: {ds.prediction_indices_to_classes(c.predict(images))}")
    
    @staticmethod
    def evaluate_accuracy(classifiers: List[cnn.Trainer], ds: DatasetWrapper, max_imgs: int = None,
                          noise_evaluation_multiplier: int = 1):
        """
        Evaluates the conventional accuracy of the supplied classifiers.
        This is done for clean images and for images corrupted witt Gaussian noise N(0, 0.8^2 I).
        :param classifiers: list of classifiers.
        :param ds: DatasetWrapper from which test images will be taken.
        :param max_imgs: limits the number of images used in evaluation (the actual number of images will be a multiple
            of batch size). If None, evaluation will be done for the full number of images produced by the test loader.
        :param noise_evaluation_multiplier: evaluation on noise-corrupted images will be done noise_evaluation_multiplier
            times.
        """
        loader = ds.get_test_loader
        if max_imgs is not None:
            loader = Util.fixed_length_loader(max_imgs, loader, False)
        for i, c in enumerate(classifiers):
            for noise_sigma in [0, 0.8]:
                accuracy, total = c.accuracy(loader, noise_sigma, noise_evaluation_multiplier)
                acc_str = f"{accuracy * 100:.2f}"
                LogUtil.info(f"Accuracy of classifier {i} on {total} validation images (noise {noise_sigma:.1f}): {acc_str:>6}%")
    
    @staticmethod
    def evaluate_conventional_robustness(classifiers: List[cnn.Trainer], ds: DatasetWrapper, max_imgs: int,
                                         l_2_bounds: List[float], l_inf_bounds: List[float]):
        """
        Evaluates the conventional robustness of the supplied classifiers as accuracy on adversarial perturbations.
        Uses L2 and L-inf norms.
        :param classifiers: list of classifiers.
        :param ds: DatasetWrapper from which test images will be taken.
        :param max_imgs: limits the number of images used in evaluation (the actual number of images will be a multiple
            of batch size). If None, evaluation will be done for the full number of images produced by the test loader.
        :param l_2_bounds: perturbation L2 norm bounds for which to perform accuracy evaluation.
        :param l_2_bounds: perturbation L-inf norm bounds for which to perform accuracy evaluation.
        """
        loader = Util.fixed_length_loader(max_imgs, ds.get_test_loader, False)
        for norm, bounds in [("scaled_l_2", l_2_bounds), ("l_inf", l_inf_bounds)]:
            for bound in bounds:
                adversary = PGDAdversary(bound, 25, 0.1, True, 0, verbose=0, norm=norm)
                for i, c in enumerate(classifiers):
                    perturb = get_conventional_perturb(c, adversary)
                    accuracy, total = c.measure_robustness(perturb, loader, ds, False)
                    acc_str = f"{accuracy * 100:.2f}"
                    LogUtil.info(f"For classifier {i}, ({norm:>10}) ║Δx║ ≤ {bound:.6f}, "
                                 f"accuracy on {total} images = {acc_str:>6}%")
                    
    @staticmethod
    def evaluate_conventional_adversarial_severity(classifiers: List[cnn.Trainer], ds: DatasetWrapper, max_imgs: int,
                                                   l_2_bound: float, l_inf_bound: float):
        """
        Evaluates the conventional adversarial severity (mean norm of minimum adversarial perturbations)
            of the supplied classifiers. Uses L2 and L-inf norms.
        :param classifiers: list of classifiers.
        :param ds: DatasetWrapper from which test images will be taken.
        :param max_imgs: limits the number of images used in evaluation (the actual number of images will be a multiple
            of batch size). If None, evaluation will be done for the full number of images produced by the test loader.
        :param l_2_bound: maximum possible L2 norm of adverarial perturbation (should be set to a number somewhat more
            than the norms of typical perturbations).
        :param l_inf_bound: maximum possible L-inf norm of adverarial perturbation (should be set to a number somewhat more
            than the norms of typical perturbations).
        """
        loader = Util.fixed_length_loader(max_imgs, ds.get_test_loader, False)
        for norm, norm_fn, bound in [("scaled_l_2", lambda x: x.norm() / np.sqrt(x.numel()), l_2_bound),
                                     ("l_inf",      lambda x: x.abs().max(),                 l_inf_bound)]:
            adversary = PGDAdversary(bound, 50, 0.05, True, 0, verbose=0, n_repeat=15, repeat_mode="min", norm=norm)
            for i, c in enumerate(classifiers):
                perturb = get_conventional_perturb(c, adversary)
                severity, std, total = c.measure_adversarial_severity(perturb, loader, ds, norm_fn, False)
                LogUtil.info(f"Adversarial severity of classifier {i} with {norm:>10} norm = {severity:.8f} "
                             f"(std = {std:.8f}, #images = {total})")
    
    @staticmethod
    def generate_images_with_classifier(classifiers: List[cnn.Trainer], ds: DatasetWrapper, no_classes: int,
                                        noise_sigma: float, norm: str, norm_bound: float,
                                        pairs_in_line: int = 3, estimation_batches: int = 100, nrow: int = 8):
        """
        Uses a classifier to generate images of each class. Starts from a mean of a number of actual images
        with added Gaussian noise. For robust classifiers, this procedure should result in images resembling
        correct class instances.
        :param classifiers: list of classifiers.
        :param ds: DatasetWrapper from which test images will be taken.
        :param no_classes: the number of classes in the dataset.
        :param noise_sigma: noise of magnitude N(0, noise_sigma^2 I) will be added to initial images.
        :param norm: a string specifying a norm as accepted by PGDAdversary.
        :param norm_bound: norm bound for PGDAdversary.
        :param pairs_in_line: pairs of images (original - generated) to present in each row.
        :param estimation_batches: number of batches to use to compute the initial mean image.
        :param nrow: nrow to be passed to ImageSet.
        """
        mean_image = torch.stack([batch.mean(dim=0) for batch, _
                                  in itertools.islice(ds.get_test_loader(), 0, estimation_batches)]).mean(dim=0)
        image_optimizer = PGDAdversary(norm_bound, 50, 0.05, False, np.infty, verbose=0, norm=norm)
        for i, classifier in enumerate(classifiers):
            LogUtil.info(f"classifier {i}:")
            classifier.set_misclassification_gradients(False)
            s = ImageSet(pairs_in_line)
            for target_label in range(no_classes):
                image = mean_image + torch.randn(*mean_image.shape) * noise_sigma
                perturb = get_conventional_perturb(classifier, image_optimizer)
                s.append([image.unsqueeze(0), perturb(image, target_label).unsqueeze(0)])
                s.maybe_show(nrow=nrow)
            s.maybe_show(True, nrow=nrow)
            classifier.set_misclassification_gradients(True)
            
    @staticmethod
    def show_reconstructed_images(gm: GenerativeModel, lines: int, images_in_line: int):
        """
        Shows several images reconstructed by the supplied generative model.
        :param gm: generative model to use.
        :param lines: lines of images to show.
        :param images_in_line: number of original - reconstructed image pairs in each line.
        """
        sampler = gm.get_sampler()
        for i in range(lines):
            images = []
            for i in range(images_in_line):
                img, _ = next(sampler)
                images += [img, gm.decode(gm.encode(img))]
            Util.imshow_tensors(*images, nrow=(images_in_line*2))
    
    @staticmethod
    def show_reconstruction_distance_statistics(gm: GenerativeModel, no_images: int):
        """
        Shows statistics on the distance of reconstructed images (based on the supplied generative model)
        from the original ones.
        :param gm: generative model to use.
        :param no_images: number of images to construct statistics for.
        """
        sampler = gm.get_sampler()
        norms, distances = np.empty(no_images), np.empty(no_images)
        norm = lambda x: x.flatten().norm() / np.sqrt(x.numel())
        for i in range(no_images):
            img, _ = next(sampler)
            norms[i] = norm(img)
            distances[i] = norm(img - gm.decode(gm.encode(img)))
        print(f"Scaled norms of original images:       mean={norms.mean():.4f}, "
              f"median={np.median(norms):.4f}, std={norms.std():.4f}")
        print(f"Scaled norms of reconstruction errors: mean={distances.mean():.4f}, "
              f"median={np.median(distances):.4f}, std={distances.std():.4f}")
        plt.figure(figsize=(10,3))
        plt.title("Scaled norms of reconstruction errors")
        plt.hist(distances)
        plt.show()
        plt.close()
    
    @staticmethod
    def show_generated_images(gm: GenerativeModel, lines: int, images_in_line: int):
        """
        Shows several images generated by the supplied generative model.
        :param gm: generative model to use.
        :param lines: lines of images to show.
        :param images_in_line: number of generated images in each line.
        """
        for i in range(lines):
            Util.imshow_tensors(gm.generate(images_in_line), nrow=images_in_line)
