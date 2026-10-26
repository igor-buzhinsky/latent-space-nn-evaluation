import argparse
from collections.abc import Iterator

import numpy as np
import scipy.stats
import torch

from latentspace.adversarial_generation import get_get_gradient
from latentspace.cnn import Trainer
from latentspace.datasets import (
    MNIST_L2_UPPER_BOUND,
    MNIST_LINF_UPPER_BOUND,
    OTHER_L2_UPPER_BOUND,
    OTHER_LINF_UPPER_BOUND,
    CelebAData,
    DatasetWrapper,
    LSUNData,
    MNISTData,
)
from latentspace.evaluation_util import EvaluationUtil
from latentspace.loaders import fixed_length_loader
from latentspace.ml_util import DatasetInfo, LogUtil, PGDAdversary
from latentspace.runtime import configure_runtime
from latentspace.torch_util import to_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trains image classifiers and evaluates them in the original image space (MNIST, CelebA, LSUN)."
    )
    parser.add_argument(
        "--command",
        required=True,
        choices=["train", "evaluate", "local_evaluate"],
        help="Command to run: train (train a classifier), evaluate (measure accuracy and robustness to conventional "
        "adversarial perturbations), or local_evaluate (experimental: for individual images, compare the error rate "
        "under Gaussian noise with the distance to the nearest adversarial example, following Ford et al., "
        "'Adversarial examples are a natural consequence of test error in noise', 2019).",
    )
    parser.add_argument(
        "--dataset", required=True, choices=["MNIST", "CelebA128Gender", "LSUN128"], help="Dataset to use."
    )
    parser.add_argument(
        "--load_filename",
        help="File with classifier weights to load. Required for evaluate and local_evaluate. "
        "For train, training continues from these weights.",
    )
    # train-specific arguments
    parser.add_argument(
        "--save_filename", help="File to save the classifier weights to after training (required for train)."
    )
    parser.add_argument(
        "--conventional_augmentation",
        action="store_true",
        help="Train with conventional data augmentation: random affine transformations and random erasing, plus "
        "flips and color jitter for color datasets (command train).",
    )
    parser.add_argument(
        "--noise_augmentation",
        action="store_true",
        help="Add Gaussian noise N(0, 0.8^2 I) to training images to improve robustness (command train).",
    )
    parser.add_argument(
        "--start_lr", type=float, default=4e-4, help="Initial learning rate (command train, default: %(default)s)."
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.25,
        help="After each epoch, the learning rate is multiplied by (1 - lr_decay) (command train, default: %(default)s).",
    )
    parser.add_argument(
        "--img_in_epoch",
        type=int,
        default=100000,
        help="Number of training images per epoch (command train, default: %(default)s).",
    )
    parser.add_argument(
        "--no_epochs", type=int, default=8, help="Number of epochs (command train, default: %(default)s)."
    )
    parser.add_argument(
        "--unit_type",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Classifier architecture: 0 = basic, 1 = with residual blocks, 2 = deeper version of 0 "
        "(default: %(default)s).",
    )
    # evaluation-specific arguments
    parser.add_argument(
        "--img_accuracy",
        type=int,
        default=10**10,
        help="Maximum number of test images for accuracy evaluation; 0 skips it (command evaluate). "
        "By default, all test images are used.",
    )
    parser.add_argument(
        "--img_robustness",
        type=int,
        default=10**10,
        help="Maximum number of test images for robustness evaluation, which measures the mean norm of minimum "
        "adversarial perturbations found with PGD; 0 skips it (command evaluate). By default, all test images are used.",
    )
    # local evaluation: checking noise / adversarial distance relationship
    parser.add_argument(
        "--local_eval_no_img",
        type=int,
        default=10,
        help="Number of test images to evaluate (command local_evaluate, default: %(default)s).",
    )
    parser.add_argument(
        "--local_eval_no_corruptions",
        type=int,
        default=2000,
        help="Number of noise corruptions per image and noise level (command local_evaluate, default: %(default)s).",
    )
    return parser


def local_evaluate(args: argparse.Namespace, c: Trainer, ds: DatasetWrapper, dataset_info: DatasetInfo) -> None:
    """
    Experimental check of ratio (1) in [Ford, N., Gilmer, J., Carlini, N., & Cubuk, D. (2019). Adversarial examples
    are a natural consequence of test error in noise].
    """
    loader = iter(ds.get_test_loader(batch_size=1))
    for i in range(args.local_eval_no_img):
        image, label = next(loader)
        if c.predict(image) != label:
            print("misclassified, skipping")
            continue
        # measure local robustness
        bound = MNIST_L2_UPPER_BOUND if dataset_info == DatasetInfo.MNIST else OTHER_L2_UPPER_BOUND

        def single_loader() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            def single_loader_() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
                yield image, label

            return single_loader_()

        adversary = PGDAdversary(bound, 50, 0.05, True, 0, verbose=0, n_repeat=15, repeat_mode="min", norm="scaled_l_2")

        def perturb(image: torch.Tensor, true_label: torch.Tensor) -> torch.Tensor:
            get_gradient = get_get_gradient(
                c, true_label, lambda x: x.view(1, *image.shape), lambda x: x, lambda x: x.view(1, -1)
            )
            return adversary.perturb(image.view(1, -1), get_gradient).view(*image.shape)

        dist = c.measure_adversarial_severity(perturb, single_loader, ds, lambda x: x.norm(), False)[0]
        # measure noise error rate
        for noise_sigma in np.linspace(0.1, 0.9, 9):
            no_corruptions = args.local_eval_no_corruptions
            error = (
                sum(
                    (
                        (c.predict(image + to_device(torch.randn(*image.shape)) * noise_sigma) != label).item()
                        for _ in range(no_corruptions)
                    )
                )
                / no_corruptions
            )
            if error >= 0.5 or np.isclose(error, 0):
                continue
            right_part = -noise_sigma * scipy.stats.norm.ppf(error)
            print(
                f"i = {i:3d}, σ = {noise_sigma:.1f} | μ = {error:.4f} | d = {dist:.5f} ?= {right_part:.5f} = -σ * Φ^{{-1}}(μ)"
            )


def main() -> None:
    configure_runtime(10500)
    LogUtil.to_pdf()

    parser = build_parser()
    args = parser.parse_args()
    if args.command in ["evaluate", "local_evaluate"] and args.load_filename is None:
        parser.error("missing argument --load_filename")
    if args.command == "train" and args.save_filename is None:
        parser.error("missing argument --save_filename")
    LogUtil.info(args)

    dataset_info = DatasetInfo[args.dataset]

    # evaluation on noise-corrupted images will be done noise_evaluation_multiplier times
    noise_evaluation_multiplier = 1

    if dataset_info == DatasetInfo.MNIST:
        trainer_name = "mnist"
        ds = MNISTData()
        no_classes = 10
    elif dataset_info == DatasetInfo.CelebA128Gender:
        trainer_name = "celeba-128"
        ds = CelebAData(20)
        no_classes = 2
    elif dataset_info == DatasetInfo.LSUN128:
        trainer_name = "lsun-128"
        ds = LSUNData()
        no_classes = 2
        # partially overcoming the problem of small (600) validation set:
        noise_evaluation_multiplier = 20
    else:
        parser.error(f"dataset {args.dataset} is not supported by this script")

    class_weights = [1 / no_classes] * no_classes
    train_loader = ds.get_train_loader if args.conventional_augmentation else ds.get_unaugmented_train_loader
    train_loader = fixed_length_loader(args.img_in_epoch, train_loader)
    c = Trainer(trainer_name, train_loader, ds.get_test_loader, args.unit_type)

    if args.command == "train":
        if args.load_filename is not None:
            c.restore_params_from_disk(args.load_filename)
        c.fit(
            class_weights,
            lr=args.start_lr,
            epochs=args.no_epochs,
            lr_decay=args.lr_decay,
            noise_sigma=(0.8 if args.noise_augmentation else 0),
        )
        c.save_params_to_disk(args.save_filename)
    elif args.command == "evaluate":
        c.restore_params_from_disk(args.load_filename)
        if dataset_info == DatasetInfo.MNIST:
            l_2_bound, l_inf_bound = MNIST_L2_UPPER_BOUND, MNIST_LINF_UPPER_BOUND
        else:
            l_2_bound, l_inf_bound = OTHER_L2_UPPER_BOUND, OTHER_LINF_UPPER_BOUND
        if args.img_accuracy > 0:
            EvaluationUtil.evaluate_accuracy([c], ds, args.img_accuracy, noise_evaluation_multiplier)
        if args.img_robustness > 0:
            EvaluationUtil.evaluate_conventional_adversarial_severity(
                [c], ds, args.img_robustness, l_2_bound, l_inf_bound
            )
    elif args.command == "local_evaluate":
        c.restore_params_from_disk(args.load_filename)
        local_evaluate(args, c, ds, dataset_info)


if __name__ == "__main__":
    main()
