import os

no_images = 10000

#unit_types = [0, 1, 2]
unit_types = [1, 2]

#datasets = ["MNIST", "CelebA128Gender", "LSUN128"]
datasets = ["LSUN128"]

for unit_type in unit_types:
    for dataset in datasets:
        filenames = " ".join([f'"classifiers_architecture{unit_type}/{training_mode}_{dataset}.bin"'
                              for training_mode in "oneepoch plain conventional robust both".split(" ")])
        print(f"*** unit_type={unit_type}, dataset={dataset}")
        os.system(f'ipython Adversarial.py -- --dataset {dataset} --no_images {no_images} '
                    f'--classifier_filenames {filenames} --search_mode both --unit_type {unit_type} '
                    f'--logdir LAT_ACC_LOG_{unit_type}_{dataset} --no_adversary --command generate_minimum')
        print()
