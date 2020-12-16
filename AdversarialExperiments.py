import os

no_images=600

#unit_types = [0, 1, 2]
unit_types = [1]
#noise_epsilons = [0.5, 1.0]
noise_epsilons = [0.5]
#datasets = ["MNIST", "CelebA128Gender", "LSUN128"]
datasets = ["CelebA128Gender"]
#commands = ["generate_minimum", "generate_bounded"]
commands = ["generate_minimum"]

for unit_type in unit_types:
    dirname = f"classifiers_architecture{unit_type}"
    for noise_epsilon in noise_epsilons:
        for dataset in datasets:
            rho = 0.3 if dataset == "MNIST" else 0.1
            for command in commands:
                print(f"*** unit_type={unit_type}, noise_epsilon={noise_epsilon}, dataset={dataset}, command={command}")
                # note: search_mode=both
                os.system(f'ipython Adversarial.py -- --dataset {dataset} --command {command} '
                          f'--no_images {no_images} --noise_epsilon {noise_epsilon} '
                          f'--classifier_filenames "{dirname}/oneepoch_{dataset}.bin" '
                          f'"{dirname}/plain_{dataset}.bin" "{dirname}/conventional_{dataset}.bin" '
                          f'"{dirname}/robust_{dataset}.bin" "{dirname}/both_{dataset}.bin" '
                          f'--search_mode both --bounded_search_rho {rho} --unit_type {unit_type} '
                          f'--logdir LOG_{unit_type}_{noise_epsilon}_{dataset}_{command}')
                print()
