import os

img_acc = 4#100000
img_rob = 600

#for unit_type in range(3):
for unit_type in [2]:
    dirname = f"classifiers_architecture{unit_type}"
    #for dataset in ["MNIST", "CelebA128Gender", "LSUN128"]:
    for dataset in ["LSUN128"]:
        for prefix in "oneepoch plain conventional robust both".split(" "):
            print(f"*** unit_type={unit_type}, dataset={dataset}, prefix={prefix}")
            os.system(f'ipython ClassifierTraining.py -- --dataset {dataset} --command evaluate '
                      f'--load_filename "{dirname}/{prefix}_{dataset}.bin" --img_accuracy {img_acc} '
                      f'--img_robustness {img_rob} --unit_type {unit_type}')
        print()