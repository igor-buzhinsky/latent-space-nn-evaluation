#!/bin/bash

if false; then
    #img_acc=100000
    img_acc=1000
    #img_rob=600
    img_rob=6

    #for dataset in MNIST CelebA128Gender LSUN128; do
    for dataset in LSUN128; do
        #for prefix in oneepoch plain conventional robust both; do
        for prefix in plain; do
            ipython ClassifierTraining.py -- --dataset $dataset --command evaluate --load_filename "new_classifiers/${prefix}_${dataset}.bin" --img_accuracy $img_acc --img_robustness $img_rob
        done
    done
fi

if true; then
    cdir=new_classifiers
    #no_images=600
    no_images=3
    for noise_epsilon in 0.5 1.0; do
        #for dataset in MNIST CelebA128Gender LSUN128; do
        for dataset in LSUN128; do
            echo noise_epsilon=$noise_epsilon dataset=$dataset
            # generate minimum perturbations:
            ipython Adversarial.py -- --dataset $dataset --command generate_minimum --no_images $no_images --noise_epsilon $noise_epsilon --classifier_filenames "$cdir/oneepoch_${dataset}.bin" "$cdir/plain_${dataset}.bin" "$cdir/conventional_${dataset}.bin" "$cdir/robust_${dataset}.bin" "$cdir/both_${dataset}.bin" --search_mode both 
            # generate bounded perturbations:
            ipython Adversarial.py -- --dataset $dataset --command generate_bounded --bounded_search_rho 0.1 --no_images $no_images --noise_epsilon $noise_epsilon --classifier_filenames "$cdir/oneepoch_${dataset}.bin" "$cdir/plain_${dataset}.bin" "$cdir/conventional_${dataset}.bin" "$cdir/robust_${dataset}.bin" "$cdir/both_${dataset}.bin" --search_mode both 
        done
    done
fi