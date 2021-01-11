#!/bin/bash

conda create --name latent-adversarial
conda activate latent-adversarial
conda install --name latent-adversarial -c pytorch -c conda-forge --file requirements.txt
pip install robustness