#!/bin/bash

dir=new_classifiers
mkdir -p "$dir"
img_in_epoch=100000
for dataset in MNIST CelebA128Gender LSUN128; do
    ipython ClassifierTraining.py -- --dataset $dataset --command train --save_filename "$dir/oneepoch_${dataset}.bin" --no_epochs 1 --img_in_epoch $img_in_epoch --start_lr 0.0004
    ipython ClassifierTraining.py -- --dataset $dataset --command train --save_filename "$dir/plain_${dataset}.bin" --no_epochs 7 --img_in_epoch $img_in_epoch --start_lr 0.0003 --load_filename "$dir/oneepoch_${dataset}.bin"
    ipython ClassifierTraining.py -- --dataset $dataset --command train --save_filename "$dir/conventional_${dataset}.bin" --img_in_epoch $img_in_epoch --conventional_augmentation
    ipython ClassifierTraining.py -- --dataset $dataset --command train --save_filename "$dir/robust_${dataset}.bin" --img_in_epoch $img_in_epoch --load_filename "$dir/plain_${dataset}.bin" --noise_augmentation
    ipython ClassifierTraining.py -- --dataset $dataset --command train --save_filename "$dir/both_${dataset}.bin" --img_in_epoch $img_in_epoch --load_filename "$dir/conventional_${dataset}.bin" --conventional_augmentation --noise_augmentation
done
