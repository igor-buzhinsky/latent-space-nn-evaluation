## Introduction

This toolset implements a framework to measure the performance of feed-forward artificial neural network classifiers with generative models.
This implementation only concerns image classification.
The framework is described in the following arXiv preprint:

* [Igor Buzhinsky, Arseny Nerinovsky, Stavros Tripakis. Metrics and methods for robustness evaluation of neural networks with generative models. arXiv preprint arXiv:2003.01993 (2020)](https://arxiv.org/abs/2003.01993)

Datasets with built-in support: [MNIST](http://yann.lecun.com/exdb/mnist/), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (gender classification), [LSUN](https://www.yf.io/p/lsun) (scene type classification: bedrooms vs. church outdoors), [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/downloads).

For CelebA and LSUN, here are some examples of approximately minimum adversarial perturbations in latent spaces for a non-robust (NR) and a robust (R) classifier:

<img src="figure.png" width="1107">

## Dependencies

To run the toolset, you need Python 3 and PyTorch. Dependencies are listed in [requirements.txt](requirements.txt). To reproduce some of our experiments with ImageNet, you will also need the [robustness](https://pypi.org/project/robustness/) package. To install all the dependencies, you may run [install.sh](install.sh). Alternatively, if you would like to minimize the size of the installation, you may try installing packages only when you run into import errors. Many packages are needed for PIONEER generative autoencoder training which you may not need to run.

## Running: MNIST

The starting point is to run the toolset on MNIST, as all trained models are small and already included into this repository.

You can work with the following command-line scripts and Jupyter notebooks:

* [Adversarial.py](Adversarial.py): calculation of latent space performance metrics, including the search of latent adversarial perturbations.
* [Adversarial.ipynb](Adversarial.ipynb): this notebook shows some features supported by Adversarial.py in a more user-friendly form. The target dataset is specified in one of the top cells.
* [ClassifierTraining.py](ClassifierTraining.py): auxiliary script that implements classifier training and evaluation of their robustness in the original space.
* [ClassifierTraining.ipynb](ClassifierTraining.ipynb): this notebook shows some features supported by ClassifierTraining.py in a more user-friendly form. The target dataset is specified in one of the top cells (CelebA and LSUN only). In addition, this notebook shows image generation with robust classifiers, which is post visible on MNIST and CelebA.
* [MNIST.ipynb](MNIST.ipynb): this notebook is the adaptation of [ClassifierTraining.ipynb](ClassifierTraining.ipynb) for MNIST. In addition, it contains code to train class-specific MNIST WGANs.

Some examples of running the aforementioned .py scripts are given in files [ClassifierTrainingExperiments.py](ClassifierTrainingExperiments.py), [ClassifierEvaluationExperiments.py](ClassifierEvaluationExperiments.py), [AdversarialExperiments.py](AdversarialExperiments.py), [LatentAccuracyExperiments.py](LatentAccuracyExperiments.py).
These scripts can also be used to reproduce the experiments from our paper.

## Running: CelebA and LSUN

CelebA and LSUN classifier models are included into the repository, but classifier training/evaluation for these datasets will require downloading these datasets. This will be done automatically for CelebA during the first run, but you will need to download LSUN on your own. You need to have the following directories:

* data/LSUN/bedroom_train_lmdb ([download archive](http://dl.yf.io/lsun/scenes/bedroom_train_lmdb.zip), size warning!)
* data/LSUN/bedroom_val_lmdb ([download archive](http://dl.yf.io/lsun/scenes/bedroom_val_lmdb.zip))
* data/LSUN/church_outdoor_train_lmdb ([download archive](http://dl.yf.io/lsun/scenes/church_outdoor_train_lmdb.zip))
* data/LSUN/church_outdoor_val_lmdb ([download archive](http://dl.yf.io/lsun/scenes/church_outdoor_val_lmdb.zip))

Evaluation of latent performance metrics for CelebA and LSUN will also require pretrained [PIONEER](https://github.com/AaltoVision/pioneer) models. You will need to download the following files (~670 MB each) and place them at the following locations:

* [pioneer/CelebA-128-female/checkpoint/20580000_state](https://drive.google.com/open?id=1X1nkyK3hkaahBYRfH36X5yyInZGYrQbW)
* [pioneer/CelebA-128-male/checkpoint/22452000_state](https://drive.google.com/open?id=1hWpm1vLXd_ay2M4AxOMxLjw4mD9gyZzH)
* [pioneer/LSUN-128-bedroom/checkpoint/15268000_state](https://drive.google.com/open?id=1sz-_3SsENJ9a4OVUF2o6q7riCcgeVlWS)
* [pioneer/LSUN-128-church_outdoor/checkpoint/20212000_state](https://drive.google.com/open?id=1eqGKeHQf-KvAe7zrvFiYjqqrEwsxvcwb)

## Running: ImageNet

To work with ImageNet-1k (1000 classes):

* Download the archive ILSVRC2012_img_val.tar with the validation data from [http://www.image-net.org/challenges/LSVRC/2012/downloads](http://www.image-net.org/challenges/LSVRC/2012/downloads) (you will need to register and accept some terms). Unpack this archive to data/ImageNet/ILSVRC2012_img_val (so that all the images are in this folder, without subfolders). The corresponding reference labels are already included in this repository.
* Download this [pretrained BigGAN model](https://drive.google.com/open?id=1dmZrcVJUAWkPBGza_XgswSuT-UODXZcO). Unpack the archive to biggan/weights/BigGAN_I128_hdf5_seed0_Gch96_Dch96_bs1_nDa8_nGa8_Glr1.0e-04_Dlr4.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema (so that .pth files are inside this folder, without subfolders). For convenience, the relevant [code of BigGAN](https://github.com/ajbrock/BigGAN-PyTorch) is already copied to this repository. If you need BiGAN on its own, use the code from the original repository.
* Download the [robust ImageNet classifier](http://andrewilyas.com/ImageNet.pt) and make it available as imagenet-models/ImageNet.pt. Other (non-robust) classifiers will be downloaded automatically upon first access.
* Check the notebook [AdversarialImageNet.ipynb](AdversarialImageNet.ipynb) or the scripts [AdversarialImageNet.py](AdversarialImageNet.py), [AdversarialImageNetExperiments.py](AdversarialImageNetExperiments.py).

## Working with other image datasets

To work with a custom dataset, you need to implement a new dataset wrapper in [latentspace/datasets.py](latentspace/datasets.py). Add support to this dataset in [latentspace/cnn.py](latentspace/cnn.py) (implement new architecture if needed). Train classifiers for this dataset with ClassifierTraining.py. Then, you will need to train generative models for each image class. Three existing options are provided by [latentspace/generative.py](latentspace/generative.py):

* class WGAN: WGANs + image reconstruction with gradient descent (Adam). This is the simplest and the most lightweight option, but this way of reconstruction for more complex datasets will take longer and may be less precise. An example of training MNIST WGANS is given in [MNIST.ipynb](MNIST.ipynb).
* class BigGAN: [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch) + image reconstruction with gradient descent (Adam). BigGAN can be replaced with other class-conditional GAN.
* class PIONEER: [PIONEER](https://github.com/AaltoVision/pioneer) autoencoder. A copy of PIONEER (slightly modified) is included into this repository ([pioneer/src](pioneer/src)), but if you need PIONEER on its own, take it from the original repository. Model training and loading is memory-intensive.

Alternatively, you can implement a different subclass of GenerativeModel. Even if you implement only generation (generate/decode) but not approximation in the latent space (encode), evaluation of latent space metrics based on generation should work.
