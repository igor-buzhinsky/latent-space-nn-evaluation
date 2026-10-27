#!/bin/bash

# the scripts and the classifier paths below are relative to the repository root
cd "$(dirname "$0")"

# classifier architectures to check: the classifiers are taken from classifiers_architecture<unit_type>
#unit_types="0 1 2"
unit_types="0"

if true; then
    #img_acc=100000
    img_acc=1000
    #img_rob=600
    img_rob=6

    #for dataset in MNIST CelebA128Gender LSUN128; do
    for unit_type in $unit_types; do
        for dataset in LSUN128; do
            #for prefix in oneepoch plain conventional robust both; do
            for prefix in plain; do
                poetry run python ClassifierTraining.py --dataset $dataset --command evaluate --load_filename "classifiers_architecture${unit_type}/${prefix}_${dataset}.bin" --img_accuracy $img_acc --img_robustness $img_rob --unit_type $unit_type
            done
        done
    done
fi

if false; then
    #num_images=600
    num_images=3
    for unit_type in $unit_types; do
        cdir=classifiers_architecture$unit_type
        for noise_epsilon in 0.5 1.0; do
            #for dataset in MNIST CelebA128Gender LSUN128; do
            for dataset in LSUN128; do
                echo unit_type=$unit_type noise_epsilon=$noise_epsilon dataset=$dataset
                # generate minimum perturbations:
                poetry run python Adversarial.py --dataset $dataset --command generate_minimum --num_images $num_images --noise_epsilon $noise_epsilon --classifier_filenames "$cdir/oneepoch_${dataset}.bin" "$cdir/plain_${dataset}.bin" "$cdir/conventional_${dataset}.bin" "$cdir/robust_${dataset}.bin" "$cdir/both_${dataset}.bin" --search_mode both --unit_type $unit_type
                # generate bounded perturbations:
                poetry run python Adversarial.py --dataset $dataset --command generate_bounded --bounded_search_rho 0.1 --num_images $num_images --noise_epsilon $noise_epsilon --classifier_filenames "$cdir/oneepoch_${dataset}.bin" "$cdir/plain_${dataset}.bin" "$cdir/conventional_${dataset}.bin" "$cdir/robust_${dataset}.bin" "$cdir/both_${dataset}.bin" --search_mode both --unit_type $unit_type
            done
        done
    done
fi