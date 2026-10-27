import shlex
import subprocess
import sys
from pathlib import Path

# scripts and relative paths below are relative to the repository root
REPO_ROOT = Path(__file__).resolve().parent

img_in_epoch = 100000

# unit_types = [0, 1, 2]
unit_types = [0]

# datasets = ["MNIST", "CelebA128Gender", "LSUN128"]
datasets = ["MNIST"]


def train(unit_type: int, dataset: str, load_prefix: str | None, save_prefix: str, more: str) -> None:
    dirname = f"classifiers_architecture{unit_type}"
    load_filename = f"{dirname}/{load_prefix}_{dataset}.bin"
    save_filename = f"{dirname}/{save_prefix}_{dataset}.bin"
    load_cmd = f'--load_filename "{load_filename}"' if load_prefix else ""
    save_cmd = f'--save_filename "{save_filename}"'
    args = (
        f"--dataset {dataset} --command train "
        f"{load_cmd} {save_cmd} --img_in_epoch {img_in_epoch} "
        f"--unit_type {unit_type} {more}"
    )
    subprocess.run([sys.executable, "ClassifierTraining.py", *shlex.split(args)], cwd=REPO_ROOT)


for unit_type in unit_types:
    (REPO_ROOT / f"classifiers_architecture{unit_type}").mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        train(unit_type, dataset, None, "oneepoch", "--num_epochs 1")
        train(unit_type, dataset, "oneepoch", "plain", "--num_epochs 7 --start_lr 0.0003")
        train(unit_type, dataset, None, "conventional", "--conventional_augmentation")
        train(unit_type, dataset, "plain", "robust", "--noise_augmentation")
        train(unit_type, dataset, "conventional", "both", "--conventional_augmentation --noise_augmentation")
