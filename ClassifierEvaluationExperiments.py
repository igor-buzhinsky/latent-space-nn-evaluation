import os

img_acc = 100000
img_rob = 600

unit_types = [0, 1, 2]
#unit_types = [0]

datasets = ["MNIST", "CelebA128Gender", "LSUN128"]
#datasets = ["LSUN128"]

for unit_type in unit_types:
    dirname = f"classifiers_architecture{unit_type}"
    for dataset in datasets:
        for prefix in "oneepoch plain conventional robust both".split(" "):
            print(f"*** unit_type={unit_type}, dataset={dataset}, prefix={prefix}")
            os.system(f'ipython ClassifierTraining.py -- --dataset {dataset} --command evaluate '
                      f'--load_filename "{dirname}/{prefix}_{dataset}.bin" --img_accuracy {img_acc} '
                      f'--img_robustness {img_rob} --unit_type {unit_type}')
        print()