import argparse
from typing import Iterator, Tuple

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
from latentspace.ml_util import DatasetInfo, LogUtil, PGDAdversary, Util


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classifier training and evaluation in the original image space.")
    parser.add_argument("--command", type=str, required=True, help="one of train, evaluate, local_evaluate")
    parser.add_argument(
        "--dataset", type=str, required=True, help="one of MNIST, CelebA128Gender, LSUN128, ImageNetAnimals"
    )
    parser.add_argument(
        "--load_filename",
        type=str,
        required=False,
        help="classifier to load for evaluation or taking pretrained parameters",
    )
    # train-specific arguments
    parser.add_argument(
        "--save_filename", type=str, required=False, help="classifier to write after training (command = train)"
    )
    parser.add_argument(
        "--conventional_augmentation", action="store_true", help="use conventional data augmentation (command = train)"
    )
    parser.add_argument(
        "--noise_augmentation", action="store_true", help="augment with N(0, 0.8^2)-noise (command = train)"
    )
    parser.add_argument("--start_lr", type=float, default=4e-4, help="initial learning rate (command = train)")
    parser.add_argument("--lr_decay", type=float, default=0.25, help="learning rate decay per epoch (command = train)")
    parser.add_argument("--img_in_epoch", type=int, default=100000, help="number of images per epoch (command = train)")
    parser.add_argument("--no_epochs", type=int, default=8, help="number of epochs (command = train)")
    parser.add_argument("--unit_type", type=int, default=0, help="architecture choice (0..2), default = 0")
    # evaluation-specific arguments
    parser.add_argument(
        "--img_accuracy",
        type=int,
        default=10**10,
        help="limit accuracy evaluation to this number of images (command = evaluate)",
    )
    parser.add_argument(
        "--img_robustness",
        type=int,
        default=10**10,
        help="limit robustness evaluation to this number of images (command = evaluate)",
    )
    # local evaluation: checking noise / adversarial distance relationship
    parser.add_argument(
        "--local_eval_no_img",
        type=int,
        default=10,
        help="number of images to perform local evaluation on (command = local_evaluate)",
    )
    parser.add_argument(
        "--local_eval_no_corruptions",
        type=int,
        default=2000,
        help="noise corruptions per each sigma (command = local_evaluate)",
    )
    return parser


def local_evaluate(args: argparse.Namespace, c: Trainer, ds: DatasetWrapper, dataset_info: DatasetInfo) -> None:
    """
    Experimental check of ratio (1) in Adversarial Examples Are a Natural Consequence of Test Error in Noise.
    """
    loader = iter(ds.get_test_loader(batch_size=1))
    for i in range(args.local_eval_no_img):
        image, label = next(loader)
        if c.predict(image) != label:
            print("misclassified, skipping")
            continue
        # measure local robustness
        bound = MNIST_L2_UPPER_BOUND if dataset_info == DatasetInfo.MNIST else OTHER_L2_UPPER_BOUND

        def single_loader() -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
            def single_loader_() -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
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
                        (
                            c.predict(image + Util.conditional_to_cuda(torch.randn(*image.shape)) * noise_sigma)
                            != label
                        ).item()
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
    Util.configure(10500)
    LogUtil.to_pdf()

    parser = build_parser()
    args = parser.parse_args()
    if args.command in ["evaluate", "local_evaluate"]:
        if args.load_filename is None:
            parser.error("missing argument --load_filename")
    elif args.command != "train":
        parser.error(f"unknown command {args.command}")
    LogUtil.info(args)

    try:
        dataset_info = DatasetInfo[args.dataset]
    except KeyError:
        parser.error(f"unsupported dataset name {args.dataset}")

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
    train_loader = Util.fixed_length_loader(args.img_in_epoch, train_loader)
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
