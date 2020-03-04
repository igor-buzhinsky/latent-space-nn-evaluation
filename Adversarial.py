#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse

from ml_util import *
import datasets
import generative
import cnn
from adversarial_generation import *
from evaluation_util import EvaluationUtil

Util.set_memory_limit(5000)
LogUtil.to_pdf()

parser = argparse.ArgumentParser(description="Generator of latent adversarial examples.")
parser.add_argument("--command", type=str, required=True,
                    help="one of test, generate_noise, generate_minimum, generate_bounded")
parser.add_argument("--dataset", type=str, required=True,
                    help="one of MNIST, CelebA128Gender, LSUN128")
parser.add_argument("--no_images", type=int, required=True,
                    help="total number of images for noise/adversarial generation")
parser.add_argument("--pgd_verbosity", type=int, default=0,
                    help="PGD verbosity: 0 = silent (default), 1, or 2 = most verbose")
parser.add_argument("--noise_perturbations_per_image", type=int, required=False,
                    help="how many noise addition sequences is done for each image (command = generate_noise)")
parser.add_argument("--noise_epsilon", type=float, required=False,
                    help="noise magnitude (positive number) for adversarial generation")
parser.add_argument("--classifier_filenames", type=str, nargs="+", required=True,
                    help="filenames with classifier models")
parser.add_argument("--search_mode", type=str, default="both",
                    help="one of 'reconstruction', 'generation', 'both' (default)")
parser.add_argument("--force_search_with_restarts", action="store_true",
                    help="to find latent perturbations, search with restarts will be used always and not only on MNIST "
                         "(command = generate_minimum)")
parser.add_argument("--no_adversary", action="store_true",
                    help="use an adversary that does nothing - this is useful to just measure latent "
                         "reconstruction/generation accuracy (this overrides --force_search_with_restarts)")
parser.add_argument("--bounded_search_rho", type=float, default=0.2,
                    help="scaled norm bound to check latent adversarial accuracy (command = generate_bounded)")
parser.add_argument("--unit_sphere_normalization", action="store_true",
                    help="(experimental, not described in the paper) search perturbations on the unit sphere instead of "
                         "the entire latent space")
args = parser.parse_args()
LogUtil.info(args)

try:
    dataset_info = DatasetInfo[args.dataset]
except KeyError:
    raise AssertionError(f"Unsupported dataset name {args.dataset}.")

class_proportions = None
if dataset_info in [DatasetInfo.CelebA128Gender, DatasetInfo.LSUN128]:
    no_classes = 2
    if dataset_info == DatasetInfo.CelebA128Gender:
        ds = datasets.CelebAData(20)
        classifier_d = "celeba-128"
        classifier_weights_filename = "celeba-128-gender-classifier/"
        class_proportions = np.array([0.583, 0.417])
    elif dataset_info == DatasetInfo.LSUN128:
        ds = datasets.LSUNData()
        classifier_d = "lsun-128"
        classifier_weights_filename = "lsun-128-classifier/"
    gm_loader = lambda label: generative.PIONEER(dataset_info, label, ds, spectral_norm_warming_no_images=25)
elif dataset_info == DatasetInfo.MNIST:
    no_classes = 10
    ds = datasets.MNISTData()
    classifier_d = "mnist"
    classifier_weights_filename = "mnist-classifier/"
    gm_loader = lambda label: generative.WGAN(dataset_info, label, ds)
else:
    raise AssertionError()
if class_proportions is None:
    class_proportions = np.repeat(1 / no_classes, no_classes)
    
def load_classifier(weights_filename: str):
    c = cnn.Trainer(classifier_d, ds.get_train_loader, ds.get_test_loader)
    c.restore_params_from_disk(weights_filename)
    return c

classifiers = [load_classifier(filename) for filename in args.classifier_filenames]

LOCAL_NOISE_EPSILONS = np.linspace(0.25, 1.00, 4)

def advgen_experiments(adversary: Adversary, total_no_images: int):
    assert args.noise_epsilon is not None, "Missing argument --noise_epsilon."
    decay_factor = EpsDTransformer().eps_to_d(args.noise_epsilon)
    if args.search_mode == "reconstruction":
        values = [False]
    elif args.search_mode == "generation":
        values = [True]
    elif args.search_mode == "both":
        values = [False, True]
    else:
        raise AssertionError("Argument --search_modes must be one of 'reconstruction', 'generation', 'both'.")
    for use_generated_images in values:
        advgen = AdversarialGenerator(None, classifiers, use_generated_images, decay_factor)
        no_images = np.round(class_proportions * total_no_images)
        for i in range(no_classes):
            LogUtil.info(f"*** {classifier_d.upper()}, CLASS {i}, "
                         f"{'GENERATED' if use_generated_images else 'RECONSTRUCTED'} ***")
            LogUtil.info(f"noise_epsilon = {args.noise_epsilon:.5f}, decay_factor = {decay_factor:.5f}")
            gm = gm_loader(i)
            advgen.set_generative_model(gm)
            advgen.generate(adversary, int(no_images[i]), True, i == 0, not args.no_adversary)
            gm.destroy()
        LogUtil.info("*** STATISTICS ***")
        advgen.print_stats(True)

if args.command == "test":
    for i in range(no_classes):
        LogUtil.info(f"*** {classifier_d.upper()}, CLASS {i} ***")
        gm = gm_loader(i)
        EvaluationUtil.show_reconstructed_images(gm, 1, 5)
        EvaluationUtil.show_generated_images(gm, 1, 10)
        # add perturbations and see classification outcomes + calculate classification accuracy
        RandomPerturbationStatistician(gm, classifiers, args.no_images, 2, True, LOCAL_NOISE_EPSILONS).process()
        # produce "class mean" vector
        advgen = AdversarialGenerator(gm, classifiers, True, 1.0)
        advgen.generate(PGDAdversary(0.01, 1, 0.1, False, 0, verbose=args.pgd_verbosity), 1, 1, False)
        gm.destroy()
elif args.command == "generate_noise":
    # measure resistance to noise
    assert args.noise_perturbations_per_image is not None, "Missing argument --noise_perturbations_per_image."
    for i in range(no_classes):
        LogUtil.info(f"*** {classifier_d.upper()}, CLASS {i} ***")
        gm = gm_loader(i)
        RandomPerturbationStatistician(gm, classifiers, args.no_images, args.noise_perturbations_per_image,
                                       True, LOCAL_NOISE_EPSILONS).process()
        gm.destroy()
elif args.command == "generate_minimum":
    # search of minimum adversarial perturbations
    max_rho = 2.5
    if args.no_adversary:
        adversary = NopAdversary()
    elif dataset_info == DatasetInfo.MNIST or args.force_search_with_restarts:
        # slow search with restarts
        adversary = PGDAdversary(max_rho, 50, 0.05, True, 0, verbose=args.pgd_verbosity, n_repeat=12, repeat_mode="min",
                                 unit_sphere_normalization=args.unit_sphere_normalization)
    else:
        # fast optimistic search
        adversary = PGDAdversary(max_rho, 1250, 0.002, False, 0, verbose=args.pgd_verbosity,
                                 unit_sphere_normalization=args.unit_sphere_normalization)
    advgen_experiments(adversary, args.no_images)
elif args.command == "generate_bounded":
    # search of bounded adversarial perturbations
    if args.no_adversary:
        adversary = NopAdversary()
    else:
        # search from different points until an adversarial example is found
        adversary = PGDAdversary(args.bounded_search_rho, 50, 0.05, True, 0, verbose=args.pgd_verbosity, n_repeat=12, repeat_mode="any",
                                 unit_sphere_normalization=args.unit_sphere_normalization)
    advgen_experiments(adversary, args.no_images)
else:
    raise RuntimeError(f"Unknown command {args.command}.")
