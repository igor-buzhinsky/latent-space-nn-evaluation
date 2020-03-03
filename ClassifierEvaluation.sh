#!/bin/bash

img_acc=100000
img_rob=600
for dataset in MNIST CelebA128Gender LSUN128; do
    for prefix in oneepoch plain conventional robust both; do
        ipython ClassifierTraining.py -- --dataset $dataset --command evaluate --load_filename "new_classifiers/${prefix}_${dataset}.bin" --img_accuracy $img_acc --img_robustness $img_rob
    done
done
