import os
from typing import *

img_in_epoch = 100000

#unit_types = [0, 1, 2]
unit_types = [0]

#datasets = ["MNIST", "CelebA128Gender", "LSUN128"]
datasets = ["ImageNetAnimals"]

for unit_type in unit_types:
    dirname = f"classifiers_architecture{unit_type}"
    os.makedirs(dirname, exist_ok=True)
    for dataset in datasets:
        def run(load_prefix: Optional[str], save_prefix: str, more: str):
            load_filename = f"{dirname}/{load_prefix}_{dataset}.bin"
            save_filename = f"{dirname}/{save_prefix}_{dataset}.bin"
            load_cmd = f'--load_filename \"{load_filename}"' if load_prefix else ""
            save_cmd = f'--save_filename \"{save_filename}"'
            os.system(f'ipython ClassifierTraining.py -- --dataset {dataset} --command train '
                      f'{load_cmd} {save_cmd} --img_in_epoch {img_in_epoch} '
                      f'--unit_type {unit_type} {more}')

        run(None,           "oneepoch",     "--no_epochs 1")
        #run("oneepoch",     "plain",        "--no_epochs 7 --start_lr 0.0003")
        run(None,           "conventional", "--conventional_augmentation")
        run("plain",        "robust",       "--noise_augmentation")
        run("conventional", "both",         "--conventional_augmentation --noise_augmentation")
        