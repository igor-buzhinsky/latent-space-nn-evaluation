#!/usr/bin/env python
# coding: utf-8

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

dataset_function = getattr(robustness_datasets, 'ImageNet')
dataset = dataset_function('./data/ImageNet')
model_kwargs = {'arch': 'resnet50', 'dataset': dataset, 'resume_path': f'./imagenet-models/ImageNet.pt'}
model, _ = model_utils.make_and_restore_model(**model_kwargs)
model = model.to("cuda:0" if Util.using_cuda else "cpu")
model.eval()

dataset_info = DatasetInfo.ImageNet
label_indices = np.arange(1000)
no_classes = len(label_indices)
classifier_d = "mnist" # any value that can be accepted by a classifier
ds = datasets.ImageNetData(label_indices)
gm = generative.BigGAN(ds, 0.25) # set built-in decay factor
class_proportions = np.repeat(1 / no_classes, no_classes)


with open("./data/ImageNet/imagenet1000_clsidx_to_labels.txt") as f: 
    imagenet_labels = f.read()
imagenet_labels = json.loads(imagenet_labels) 


class Resize(torch.nn.Module):
    def __init__(self, side: int):
        super().__init__()
        self.side = side
    
    def forward(self, x: torch.Tensor):
        return torch.nn.functional.interpolate(x, size=(self.side, self.side),
                                               mode="bicubic", align_corners=False)

modelzoo_normalize_transform = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

def normalize_modelzoo(x):
    # [-1, 1] -> [0, 1], then normalize
    # https://pytorch.org/docs/stable/torchvision/models.html
    return modelzoo_normalize_transform((x[0] + 1) / 2).unsqueeze(0)

def normalize_01(x):
    # [-1, 1] -> [0, 1]
    return (x + 1) / 2
    
class RobustnessClassifierWrapper(Trainer):
    def __init__(self, model):
        super().__init__(classifier_d, ds.get_train_loader, ds.get_test_loader, unit_type=0)
        self.model = torch.nn.Sequential(
            Resize(224),
            Lambda(normalize_01),
            model,
            Lambda(lambda x: x[0]),
        )
        
robust_classifiers = [RobustnessClassifierWrapper(model)]


class ModelZooClassifierWrapper(Trainer):
    def __init__(self, model, side: int):
        super().__init__(classifier_d, ds.get_train_loader, ds.get_test_loader, unit_type=0)
        self.model = torch.nn.Sequential(
            Resize(side),
            Lambda(normalize_modelzoo),
            model,
        )

nonrobust_classifiers = [(models.squeezenet1_0, 256),   # error 41.90
                         (models.alexnet, 256),         # error 43.45
                         (models.resnet18, 224),        # error 30.24
                         (models.resnext50_32x4d, 224)] # error 21.49
nonrobust_classifiers = [ModelZooClassifierWrapper(Util.conditional_to_cuda(m(pretrained=True)), side)
                         for m, side in nonrobust_classifiers]
classifiers = nonrobust_classifiers + robust_classifiers


def advgen_experiments(adversary: Adversary, noise_eps: float, total_images: int):
    decay_factor = EpsDTransformer().eps_to_d(noise_eps)
    label_printer = lambda x: str(x.item()) #+ " " + imagenet_labels[str(x.item())]
    advgen = AdversarialGenerator(None, classifiers, True, decay_factor, label_printer)
    advgen.set_generative_model(gm)
    for i in np.random.choice(label_indices, size=total_images):
        no_images = 1
        LogUtil.info(f"*** CLASS {i}: {imagenet_labels[str(i)]} ***")
        gm.configure_label(i)
        advgen.generate(adversary, no_images, False, clear_stat=(i == 0))
    LogUtil.info("*** STATISTICS ***")
    advgen.print_stats(True)


# ### Measure latent adversarial accuracy (LGA)
# advgen_experiments(NopAdversary(), noise_eps=1.0, total_images=100)


# ### Measure LAGS
if True:
    max_rho, noise_eps = 2.5, 1.0
    #max_rho, noise_eps = 2.5, 0.5
    adversary = PGDAdversary(max_rho, 50, 0.05, False, 0, verbose=0, n_repeat=12, repeat_mode="min")
    #adversary = PGDAdversary(max_rho, 1250, 0.002, False, 0, verbose=0)
    advgen_experiments(adversary, noise_eps, total_images=600)

# Measure LAGA
if False:
    rho, noise_eps = 0.3, 1.0
    # search with restarts (a sequence of restarts will terminate if an adversarial perturbation is found)
    adversary = PGDAdversary(rho, 50, 0.05, True, 0, verbose=0, n_repeat=12, repeat_mode="any")
    advgen_experiments(adversary, noise_eps, total_images=100)

