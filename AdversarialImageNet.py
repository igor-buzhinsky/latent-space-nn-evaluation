#!/usr/bin/env python
# coding: utf-8

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from robustness import model_utils, datasets as robustness_datasets

from latentspace.ml_util import *
import latentspace.datasets as datasets
import latentspace.generative as generative
import latentspace.cnn as cnn
from latentspace.adversarial_generation import *
from latentspace.evaluation_util import EvaluationUtil

Util.configure(6000)
LogUtil.to_pdf()

parser = argparse.ArgumentParser(description="Generator of latent adversarial examples for ImageNet.")
parser.add_argument("--command", type=str, required=True,
                    help="one of generate_minimum, generate_bounded, measure_accuracy, "
                         "generate_conventional_from_gan, generate_conventional_from_validation")
parser.add_argument("--no_images", type=int, required=True,
                    help="total number of images for adversarial generation or accuracy evaluation")
parser.add_argument("--bounded_search_rho", type=float, default=0.2,
                    help="scaled norm bound to check latent adversarial accuracy "
                         "(command = generate_bounded), default = 0.2")
parser.add_argument("--noise_epsilon", type=float, required=False, default=1.0,
                    help="noise magnitude (positive number) for adversarial generation, default = 1.0")
parser.add_argument("--no_adversary", action="store_true",
                    help="use an adversary that does nothing - this is useful to just measure latent "
                         "reconstruction/generation accuracy")
parser.add_argument("--logdir", type=str, default=None,
                    help="set a custom logging directory and remove its previous contents (by default, a new "
                         " name will be generated based on the timestamp)")
args = parser.parse_args()

if args.logdir is not None:
    LogUtil.set_custom_dirname(args.logdir)
LogUtil.info(args)

# configure the generative model (BigGAN)
NO_LABELS = 1000
dataset_info = DatasetInfo.ImageNet
label_indices = np.arange(NO_LABELS)
no_classes = len(label_indices)
classifier_d = "none"
ds = datasets.ImageNetData()
gm = generative.BigGAN(ds, 0.25) # set built-in decay factor

# load the robust classifier
dataset = robustness_datasets.ImageNet("./data/ImageNet")
model_kwargs = {"arch": "resnet50", "dataset": dataset, "resume_path": f"./imagenet-models/ImageNet.pt"}
model, _ = model_utils.make_and_restore_model(**model_kwargs)
model = model.to("cuda:0" if Util.using_cuda else "cpu")
model.eval()

# normalize input + wrap the robust classifier
class RobustnessClassifierWrapper(Trainer):
    def __init__(self, model):
        super().__init__(classifier_d, ds.get_train_loader, ds.get_test_loader, unit_type=0)
        self.model = torch.nn.Sequential(
            Resize(224),
            Lambda(lambda x: (x + 1) / 2),
            model,
            Lambda(lambda x: x[0]),
        )
        
robust_classifiers = [RobustnessClassifierWrapper(model)]

# load non-robust classifiers, then normalizae input + wrap
class ModelZooClassifierWrapper(Trainer):
    modelzoo_normalize_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    @staticmethod
    def normalize_modelzoo(x):
        # [-1, 1] -> [0, 1], then normalize
        # https://pytorch.org/docs/stable/torchvision/models.html
        return ModelZooClassifierWrapper.modelzoo_normalize_transform((x[0] + 1) / 2).unsqueeze(0)
    
    def __init__(self, model, side: int):
        super().__init__(classifier_d, ds.get_train_loader, ds.get_test_loader, unit_type=0)
        self.model = torch.nn.Sequential(
            Resize(side),
            Lambda(ModelZooClassifierWrapper.normalize_modelzoo),
            model,
        )

nonrobust_classifiers = [(models.squeezenet1_0, 256),   # ImageNet top-1 error 41.90
                         (models.alexnet, 256),         # ImageNet top-1 error 43.45
                         (models.resnet18, 224),        # ImageNet top-1 error 30.24
                         (models.resnext50_32x4d, 224)] # ImageNet top-1 error 21.49
nonrobust_classifiers = [ModelZooClassifierWrapper(Util.conditional_to_cuda(m(pretrained=True)), side)
                         for m, side in nonrobust_classifiers]

classifiers = nonrobust_classifiers + robust_classifiers


def get_target_labels(total_images: int) -> np.ndarray:
    if total_images % NO_LABELS == 0:
        # use stratified labels
        return np.repeat(np.arange(NO_LABELS), total_images // NO_LABELS)
    else:
        # use totally random labels
        return np.random.choice(label_indices, size=total_images) 

def advgen_experiments(adversary: Adversary, total_images: int):
    decay_factor = EpsDTransformer().eps_to_d(args.noise_epsilon)
    advgen = AdversarialGenerator(None, classifiers, True, decay_factor)
    advgen.set_generative_model(gm)
    label_array = get_target_labels(total_images)   
    for label in label_array:
        LogUtil.info(f"*** CLASS {label}: {ds.printed_classes[label]} ***")
        gm.configure_label(label)
        advgen.generate(adversary, 1, True, False, not args.no_adversary)
    LogUtil.info("*** STATISTICS ***")
    advgen.print_stats(plot=(not args.no_adversary), print_norm_statistics=(not args.no_adversary))


if args.command == "generate_minimum":
    # Measure LAGS
    max_rho = 2.5
    if args.no_adversary:
        adversary = NopAdversary()
    else:
        adversary = PGDAdversary(max_rho, 50, 0.05, False, 0, verbose=0, n_repeat=12, repeat_mode="min")
    advgen_experiments(adversary, total_images=args.no_images)
elif args.command == "generate_bounded":
    # Measure LAGA
    rho = args.bounded_search_rho
    if args.no_adversary:
        adversary = NopAdversary()
    else:
        adversary = PGDAdversary(rho, 50, 0.05, True, 0, verbose=0, n_repeat=12, repeat_mode="any")
    advgen_experiments(adversary, total_images=args.no_images)
elif args.command == "measure_accuracy":
    loader = Util.fixed_length_loader(args.no_images, ds.get_test_loader, False)
    for i, c in enumerate(classifiers):
        accuracy, total = c.accuracy(loader, 0, 1)
        acc_str = f"{accuracy * 100:.2f}"
        LogUtil.info(f"Accuracy of classifier {i} on {total} validation images: {acc_str:>6}%")
elif args.command in ["generate_conventional_from_gan", "generate_conventional_from_validation"]:
    if args.command == "generate_conventional_from_gan":
        def loader():
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
        loader = Util.fixed_length_loader(args.no_images, ds.get_test_loader, False)
    norms = [
        ("scaled_l_2", lambda x: x.norm() / np.sqrt(x.numel()), datasets.OTHER_L2_UPPER_BOUND),
        ("l_inf",      lambda x: x.abs().max(),                 datasets.OTHER_LINF_UPPER_BOUND)
    ]
    for norm, norm_fn, bound in norms:
        adversary = PGDAdversary(bound, 50, 0.05, True, 0, verbose=0, n_repeat=15, repeat_mode="min", norm=norm)
        for i, c in enumerate(classifiers):
            perturb = get_conventional_perturb(c, adversary)
            severity, std, total = c.measure_adversarial_severity(perturb, loader, ds, norm_fn, False)
            LogUtil.info(f"Adversarial severity of classifier {i} with {norm:>10} norm = {severity:.8f} "
                         f"(std = {std:.8f}, #images = {total})")
else:
    raise RuntimeError(f"Unknown command {args.command}.")
