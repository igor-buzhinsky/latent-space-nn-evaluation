import argparse
from collections.abc import Callable

import numpy as np

import latentspace.cnn as cnn
import latentspace.datasets as datasets
import latentspace.generative as generative
from latentspace.adversarial_generation import AdversarialGenerator, EpsDTransformer, RandomPerturbationStatistician
from latentspace.adversaries import Adversary, NopAdversary, PGDAdversary
from latentspace.datasets import DatasetInfo
from latentspace.evaluation_util import show_generated_images, show_reconstructed_images
from latentspace.logging_util import LogUtil
from latentspace.runtime import raise_open_file_limit, set_memory_limit

# values of --search_mode and the corresponding values of use_generated_images
SEARCH_MODES = {"reconstruction": [False], "generation": [True], "both": [False, True]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluates image classifiers with latent space performance metrics computed with generative models "
        "(MNIST, CelebA, LSUN)."
    )
    parser.add_argument(
        "--command",
        required=True,
        choices=["test", "generate_noise", "generate_minimum", "generate_bounded"],
        help="Command to run: test (show reconstructed and generated images and run quick checks), generate_noise "
        "(measure accuracy under random latent noise), generate_minimum (search for minimum latent adversarial "
        "perturbations), or generate_bounded (measure latent adversarial accuracy within the norm bound "
        "--bounded_search_rho).",
    )
    parser.add_argument(
        "--dataset", required=True, choices=["MNIST", "CelebA128Gender", "LSUN128"], help="Dataset to use."
    )
    parser.add_argument(
        "--num_images",
        type=int,
        required=True,
        help="Number of images to process. For generate_minimum and generate_bounded, this is the total number, split "
        "across classes in proportion to class frequencies. For test and generate_noise, this is the number per class.",
    )
    parser.add_argument(
        "--pgd_verbosity",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Verbosity of the PGD search: 0 = silent, 1, or 2 = most verbose (default: %(default)s).",
    )
    parser.add_argument(
        "--noise_perturbations_per_image",
        type=int,
        help="Number of random latent perturbations per image and noise magnitude (required for generate_noise).",
    )
    parser.add_argument(
        "--noise_epsilon",
        type=float,
        default=1.0,
        help="Noise magnitude epsilon (positive number): before the adversarial search, latent codes are multiplied "
        "by 1 / sqrt(1 + epsilon^2) (commands generate_minimum and generate_bounded, default: %(default)s).",
    )
    parser.add_argument(
        "--classifier_filenames",
        nargs="+",
        required=True,
        help="One or more files with classifier weights, e.g. classifiers_architecture0/robust_MNIST.bin. "
        "Their architecture must match --unit_type.",
    )
    parser.add_argument(
        "--search_mode",
        choices=list(SEARCH_MODES),
        default="both",
        help="Which latent codes to search from: reconstruction (codes of real images obtained by the generative "
        "model's encoder), generation (random codes), or both (commands generate_minimum and generate_bounded, "
        "default: %(default)s).",
    )
    parser.add_argument(
        "--force_search_with_restarts",
        action="store_true",
        help="Always use the slower search with restarts to find minimum latent perturbations, not only on MNIST "
        "(command generate_minimum).",
    )
    parser.add_argument(
        "--no_adversary",
        action="store_true",
        help="Use an adversary that does nothing. This is useful to only measure the accuracy of classifiers on "
        "reconstructed or generated images. Overrides --force_search_with_restarts.",
    )
    parser.add_argument(
        "--bounded_search_rho",
        type=float,
        default=0.2,
        help="Scaled norm bound for latent perturbations when measuring latent adversarial accuracy "
        "(command generate_bounded, default: %(default)s).",
    )
    parser.add_argument(
        "--unit_type",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Classifier architecture: 0 = basic, 1 = with residual blocks, 2 = deeper version of 0 "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--unit_sphere_normalization",
        action="store_true",
        help="Experimental, not described in the paper: search for perturbations on the unit sphere instead of the "
        "entire latent space.",
    )
    parser.add_argument(
        "--logdir",
        help="Directory for logs and figures. Its previous contents are deleted. By default, a new directory named "
        "after the current timestamp is created.",
    )
    return parser


LOCAL_NOISE_EPSILONS = np.linspace(0.25, 1.00, 4)


def load_classifier(
    weights_filename: str, classifier_d: str, ds: datasets.DatasetWrapper, unit_type: int
) -> cnn.Trainer:
    c = cnn.Trainer(classifier_d, ds.get_train_loader, ds.get_test_loader, unit_type)
    c.restore_params_from_disk(weights_filename)
    return c


def advgen_experiments(
    args: argparse.Namespace,
    adversary: Adversary,
    classifiers: list[cnn.Trainer],
    gm_loader: Callable[[int], generative.GenerativeModel],
    class_proportions: np.ndarray,
    classifier_d: str,
) -> None:
    decay_factor = EpsDTransformer().eps_to_d(args.noise_epsilon)
    num_classes = len(class_proportions)
    for use_generated_images in SEARCH_MODES[args.search_mode]:
        advgen = AdversarialGenerator(None, classifiers, use_generated_images, decay_factor)
        num_images = np.round(class_proportions * args.num_images)
        for i in range(num_classes):
            LogUtil.info(
                f"*** {classifier_d.upper()}, CLASS {i}, "
                f"{'GENERATED' if use_generated_images else 'RECONSTRUCTED'} ***"
            )
            LogUtil.info(f"noise_epsilon = {args.noise_epsilon:.5f}, decay_factor = {decay_factor:.5f}")
            gm = gm_loader(i)
            advgen.set_generative_model(gm)
            advgen.generate(adversary, int(num_images[i]), True, i == 0, not args.no_adversary)
            gm.destroy()
        LogUtil.info("*** STATISTICS ***")
        advgen.print_stats(plot=(not args.no_adversary), print_norm_statistics=(not args.no_adversary))


def main() -> None:
    set_memory_limit(10000)
    raise_open_file_limit()
    LogUtil.to_pdf()

    parser = build_parser()
    args = parser.parse_args()
    if args.command in ["generate_minimum", "generate_bounded"]:
        if args.noise_epsilon is None:
            parser.error("missing argument --noise_epsilon")
    elif args.command == "generate_noise":
        if args.noise_perturbations_per_image is None:
            parser.error("missing argument --noise_perturbations_per_image")

    if args.logdir is not None:
        LogUtil.set_custom_dirname(args.logdir)
    LogUtil.info(args)

    dataset_info = DatasetInfo[args.dataset]

    if dataset_info in [DatasetInfo.CelebA128Gender, DatasetInfo.LSUN128]:
        num_classes = 2
        if dataset_info == DatasetInfo.CelebA128Gender:
            ds = datasets.CelebAData(20)
            classifier_d = "celeba-128"
            class_proportions = np.array([0.583, 0.417])
        else:
            ds = datasets.LSUNData()
            classifier_d = "lsun-128"
            class_proportions = np.repeat(1 / num_classes, num_classes)

        def gm_loader(label: int) -> generative.GenerativeModel:
            return generative.PIONEER(dataset_info, label, ds, spectral_norm_warming_num_images=25)

    elif dataset_info == DatasetInfo.MNIST:
        num_classes = 10
        ds = datasets.MNISTData()
        classifier_d = "mnist"
        class_proportions = np.repeat(1 / num_classes, num_classes)

        def gm_loader(label: int) -> generative.GenerativeModel:
            return generative.WGAN(dataset_info, label, ds)

    else:
        parser.error(f"dataset {args.dataset} is not supported by this script")

    classifiers = [
        load_classifier(filename, classifier_d, ds, args.unit_type) for filename in args.classifier_filenames
    ]

    if args.command == "test":
        for i in range(num_classes):
            LogUtil.info(f"*** {classifier_d.upper()}, CLASS {i} ***")
            gm = gm_loader(i)
            show_reconstructed_images(gm, 1, 5)
            show_generated_images(gm, 1, 10)
            # add perturbations and see classification outcomes + calculate classification accuracy
            RandomPerturbationStatistician(gm, classifiers, args.num_images, 2, True, LOCAL_NOISE_EPSILONS).process()
            # produce "class mean" vector
            advgen = AdversarialGenerator(gm, classifiers, True, 1.0)
            advgen.generate(PGDAdversary(0.01, 1, 0.1, False, 0, verbose=args.pgd_verbosity), 1, True, False)
            gm.destroy()
    elif args.command == "generate_noise":
        # measure resistance to noise
        for i in range(num_classes):
            LogUtil.info(f"*** {classifier_d.upper()}, CLASS {i} ***")
            gm = gm_loader(i)
            RandomPerturbationStatistician(
                gm, classifiers, args.num_images, args.noise_perturbations_per_image, True, LOCAL_NOISE_EPSILONS
            ).process()
            gm.destroy()
    elif args.command == "generate_minimum":
        # search of minimum adversarial perturbations
        max_rho = 2.5
        if args.no_adversary:
            adversary = NopAdversary()
        elif dataset_info == DatasetInfo.MNIST or args.force_search_with_restarts:
            # slow search with restarts
            adversary = PGDAdversary(
                max_rho,
                50,
                0.05,
                True,
                0,
                verbose=args.pgd_verbosity,
                n_repeat=12,
                repeat_mode="min",
                unit_sphere_normalization=args.unit_sphere_normalization,
            )
        else:
            # fast optimistic search
            adversary = PGDAdversary(
                max_rho,
                1250,
                0.002,
                False,
                0,
                verbose=args.pgd_verbosity,
                unit_sphere_normalization=args.unit_sphere_normalization,
            )
        advgen_experiments(args, adversary, classifiers, gm_loader, class_proportions, classifier_d)
    elif args.command == "generate_bounded":
        # search of bounded adversarial perturbations
        if args.no_adversary:
            adversary = NopAdversary()
        else:
            # search from different points until an adversarial example is found
            adversary = PGDAdversary(
                args.bounded_search_rho,
                50,
                0.05,
                True,
                0,
                verbose=args.pgd_verbosity,
                n_repeat=12,
                repeat_mode="any",
                unit_sphere_normalization=args.unit_sphere_normalization,
            )
        advgen_experiments(args, adversary, classifiers, gm_loader, class_proportions, classifier_d)


if __name__ == "__main__":
    main()
