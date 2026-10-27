import shlex
import subprocess
import sys
from pathlib import Path

# scripts and relative paths below are relative to the repository root
REPO_ROOT = Path(__file__).resolve().parent

num_images = 10000

# unit_types = [0, 1, 2]
unit_types = [1, 2]

# datasets = ["MNIST", "CelebA128Gender", "LSUN128"]
datasets = ["LSUN128"]

for unit_type in unit_types:
    for dataset in datasets:
        filenames = " ".join(
            [
                f'"classifiers_architecture{unit_type}/{training_mode}_{dataset}.bin"'
                for training_mode in ["oneepoch", "plain", "conventional", "robust", "both"]
            ]
        )
        print(f"*** unit_type={unit_type}, dataset={dataset}")
        args = (
            f"--dataset {dataset} --num_images {num_images} "
            f"--classifier_filenames {filenames} --search_mode both --unit_type {unit_type} "
            f"--logdir LAT_ACC_LOG_{unit_type}_{dataset} --no_adversary --command generate_minimum"
        )
        subprocess.run([sys.executable, "Adversarial.py", *shlex.split(args)], cwd=REPO_ROOT)
        print()
