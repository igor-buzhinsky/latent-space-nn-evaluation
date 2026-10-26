import argparse
from collections.abc import Iterator

import numpy as np
import torch
import torchvision
import torchvision.models as models

import latentspace.datasets as datasets
import latentspace.generative as generative
from latentspace.adversarial_generation import AdversarialGenerator, EpsDTransformer, get_conventional_perturb
from latentspace.adversaries import Adversary, NopAdversary, PGDAdversary
from latentspace.cnn import Trainer
from latentspace.compat import patch_torchvision_for_robustness
from latentspace.layers import Lambda, Resize
from latentspace.loaders import fixed_length_loader
from latentspace.logging_util import LogUtil
from latentspace.runtime import configure_runtime
from latentspace.torch_util import DEVICE, to_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluates ImageNet classifiers with latent space performance metrics computed with BigGAN, "
        "and with conventional metrics."
    )
    parser.add_argument(
        "--command",
        required=True,
        choices=[
            "generate_minimum",
            "generate_bounded",
            "measure_accuracy",
            "generate_conventional_from_gan",
            "generate_conventional_from_validation",
        ],
        help="Command to run: generate_minimum (search for minimum latent adversarial perturbations), "
        "generate_bounded (measure latent adversarial accuracy within the norm bound --bounded_search_rho), "
        "measure_accuracy (accuracy on validation images), generate_conventional_from_gan or "
        "generate_conventional_from_validation (robustness to conventional adversarial perturbations of "
        "BigGAN-generated or validation images).",
    )
    parser.add_argument(
        "--no_images",
        type=int,
        required=True,
        help="Total number of images to process. For BigGAN-generated images, if this is a multiple of 1000, "
        "each class is used equally often; otherwise, classes are chosen randomly.",
    )
    parser.add_argument(
        "--bounded_search_rho",
        type=float,
        default=0.2,
        help="Scaled norm bound for latent perturbations when measuring latent adversarial accuracy "
        "(command generate_bounded, default: %(default)s).",
    )
    parser.add_argument(
        "--noise_epsilon",
        type=float,
        default=1.0,
        help="Noise magnitude epsilon (positive number): before the adversarial search, latent codes are multiplied "
        "by 1 / sqrt(1 + epsilon^2) (commands generate_minimum and generate_bounded, default: %(default)s).",
    )
    parser.add_argument(
        "--no_adversary",
        action="store_true",
        help="Use an adversary that does nothing. This is useful to only measure the accuracy of classifiers on "
        "generated images.",
    )
    parser.add_argument(
        "--logdir",
        help="Directory for logs and figures. Its previous contents are deleted. By default, a new directory named "
        "after the current timestamp is created.",
    )
    return parser


NO_LABELS = 1000
# classifiers are not trained in this script, so no built-in architecture is needed
CLASSIFIER_D = "none"


# normalize input + wrap the robust classifier
class RobustnessClassifierWrapper(Trainer):
    def __init__(self, model: torch.nn.Module, ds: datasets.DatasetWrapper) -> None:
        super().__init__(CLASSIFIER_D, ds.get_train_loader, ds.get_test_loader, unit_type=0)
        self.model = torch.nn.Sequential(
            Resize(224),
            Lambda(lambda x: (x + 1) / 2),
            model,
            Lambda(lambda x: x[0]),
        )


# normalize input + wrap a non-robust classifier
class ModelZooClassifierWrapper(Trainer):
    modelzoo_normalize_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    @staticmethod
    def normalize_modelzoo(x: torch.Tensor) -> torch.Tensor:
        # [-1, 1] -> [0, 1], then normalize
        # https://pytorch.org/docs/stable/torchvision/models.html
        return ModelZooClassifierWrapper.modelzoo_normalize_transform((x[0] + 1) / 2).unsqueeze(0)

    def __init__(self, model: torch.nn.Module, side: int, ds: datasets.DatasetWrapper) -> None:
        super().__init__(CLASSIFIER_D, ds.get_train_loader, ds.get_test_loader, unit_type=0)
        self.model = torch.nn.Sequential(
            Resize(side),
            Lambda(ModelZooClassifierWrapper.normalize_modelzoo),
            model,
        )


def load_classifiers(ds: datasets.DatasetWrapper) -> list[Trainer]:
    # load the robust classifier
    patch_torchvision_for_robustness()
    from robustness import datasets as robustness_datasets
    from robustness import model_utils

    dataset = robustness_datasets.ImageNet("./data/ImageNet")
    model_kwargs = {"arch": "resnet50", "dataset": dataset, "resume_path": "./imagenet-models/ImageNet.pt"}
    model, _ = model_utils.make_and_restore_model(**model_kwargs)
    model = model.to(DEVICE)
    model.eval()
    robust_classifiers = [RobustnessClassifierWrapper(model, ds)]

    # load non-robust classifiers
    nonrobust_models = [
        (models.squeezenet1_0, 256),  # ImageNet top-1 error 41.90
        (models.alexnet, 256),  # ImageNet top-1 error 43.45
        (models.resnet18, 224),  # ImageNet top-1 error 30.24
        (models.resnext50_32x4d, 224),
    ]  # ImageNet top-1 error 21.49
    nonrobust_classifiers = [
        ModelZooClassifierWrapper(to_device(m(pretrained=True)), side, ds) for m, side in nonrobust_models
    ]
    classifiers: list[Trainer] = [*nonrobust_classifiers, *robust_classifiers]
    return classifiers


def get_target_labels(total_images: int) -> np.ndarray:
    if total_images % NO_LABELS == 0:
        # use stratified labels
        return np.repeat(np.arange(NO_LABELS), total_images // NO_LABELS)
    # use totally random labels
    return np.random.choice(np.arange(NO_LABELS), size=total_images)


def advgen_experiments(
    args: argparse.Namespace,
    adversary: Adversary,
    classifiers: list[Trainer],
    gm: generative.BigGAN,
    ds: datasets.ImageNetData,
) -> None:
    decay_factor = EpsDTransformer().eps_to_d(args.noise_epsilon)
    advgen = AdversarialGenerator(None, classifiers, True, decay_factor)
    advgen.set_generative_model(gm)
    label_array = get_target_labels(args.no_images)
    for label in label_array:
        LogUtil.info(f"*** CLASS {label}: {ds.printed_classes[label]} ***")
        gm.configure_label(label)
        advgen.generate(adversary, 1, True, False, not args.no_adversary)
    LogUtil.info("*** STATISTICS ***")
    advgen.print_stats(plot=(not args.no_adversary), print_norm_statistics=(not args.no_adversary))


def conventional_experiments(
    args: argparse.Namespace, classifiers: list[Trainer], gm: generative.BigGAN, ds: datasets.ImageNetData
) -> None:
    if args.command == "generate_conventional_from_gan":

        def loader() -> Iterator[tuple[list[torch.Tensor], list[int]]]:
            label_array = get_target_labels(args.no_images)
            images, labels = [], []
            for label in label_array:
                gm.configure_label(label)
                images += [gm.generate(1, detach=True)[0]]
                labels += [label]
                if len(images) == datasets.DatasetWrapper.test_batch_size:
                    yield images, labels
                    images, labels = [], []

    else:
        loader = fixed_length_loader(args.no_images, ds.get_test_loader, False)
    norms = [
        ("scaled_l_2", lambda x: x.norm() / np.sqrt(x.numel()), datasets.OTHER_L2_UPPER_BOUND),
        ("l_inf", lambda x: x.abs().max(), datasets.OTHER_LINF_UPPER_BOUND),
    ]
    for norm, norm_fn, bound in norms:
        adversary = PGDAdversary(bound, 50, 0.05, True, 0, verbose=0, n_repeat=15, repeat_mode="min", norm=norm)
        for i, c in enumerate(classifiers):
            perturb = get_conventional_perturb(c, adversary)
            severity, std, total = c.measure_adversarial_severity(perturb, loader, ds, norm_fn, False)
            LogUtil.info(
                f"Adversarial severity of classifier {i} with {norm:>10} norm = {severity:.8f} "
                f"(std = {std:.8f}, #images = {total})"
            )


def main() -> None:
    configure_runtime(11000)
    LogUtil.to_pdf()

    parser = build_parser()
    args = parser.parse_args()
    if args.logdir is not None:
        LogUtil.set_custom_dirname(args.logdir)
    LogUtil.info(args)

    # configure the generative model (BigGAN)
    ds = datasets.ImageNetData()
    gm = generative.BigGAN(ds, 0.25)  # set built-in decay factor
    classifiers = load_classifiers(ds)

    if args.command == "generate_minimum":
        # Measure LAGS
        max_rho = 2.5
        if args.no_adversary:
            adversary = NopAdversary()
        else:
            adversary = PGDAdversary(max_rho, 50, 0.05, False, 0, verbose=0, n_repeat=12, repeat_mode="min")
        advgen_experiments(args, adversary, classifiers, gm, ds)
    elif args.command == "generate_bounded":
        # Measure LAGA
        rho = args.bounded_search_rho
        if args.no_adversary:
            adversary = NopAdversary()
        else:
            adversary = PGDAdversary(rho, 50, 0.05, True, 0, verbose=0, n_repeat=12, repeat_mode="any")
        advgen_experiments(args, adversary, classifiers, gm, ds)
    elif args.command == "measure_accuracy":
        loader = fixed_length_loader(args.no_images, ds.get_test_loader, False)
        for i, c in enumerate(classifiers):
            accuracy, total = c.accuracy(loader, 0, 1)
            acc_str = f"{accuracy * 100:.2f}"
            LogUtil.info(f"Accuracy of classifier {i} on {total} validation images: {acc_str:>6}%")
    else:
        conventional_experiments(args, classifiers, gm, ds)


if __name__ == "__main__":
    main()
