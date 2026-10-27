"""
Smoke test: searches for latent adversarial perturbations against bundled classifiers on a real dataset with
Adversarial.py, using few images.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

# scripts and relative paths below are relative to the repository root
REPO_ROOT = Path(__file__).resolve().parent

DATASETS = ["MNIST", "CelebA128Gender", "LSUN128"]
PREFIXES = ["oneepoch", "plain", "conventional", "robust", "both"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=["CelebA128Gender"],
        help="Datasets to check (default: %(default)s).",
    )
    parser.add_argument(
        "--unit_types",
        nargs="+",
        type=int,
        choices=[0, 1, 2],
        default=[0],
        help="Classifier architectures to check; classifiers are taken from classifiers_architecture<unit_type> "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--noise_epsilons",
        nargs="+",
        type=float,
        default=[0.5, 1.0],
        help="Values of --noise_epsilon to check (default: %(default)s).",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Total number of images, split across classes (default: %(default)s; full evaluation uses 600).",
    )
    return parser


def run(args: str) -> None:
    print(f"*** Adversarial.py {args}", flush=True)
    result = subprocess.run([sys.executable, "Adversarial.py", *shlex.split(args)], cwd=REPO_ROOT)
    if result.returncode != 0:
        # the error has already been reported by the script; stop at the first failure
        sys.exit(result.returncode)


def main() -> None:
    args = build_parser().parse_args()
    for unit_type in args.unit_types:
        classifier_dir = f"classifiers_architecture{unit_type}"
        for noise_epsilon in args.noise_epsilons:
            for dataset in args.datasets:
                classifiers = " ".join(f'"{classifier_dir}/{prefix}_{dataset}.bin"' for prefix in PREFIXES)
                common = (
                    f"--dataset {dataset} --num_images {args.num_images} --noise_epsilon {noise_epsilon} "
                    f"--classifier_filenames {classifiers} --search_mode both --unit_type {unit_type}"
                )
                # minimum perturbations
                run(f"--command generate_minimum {common}")
                # bounded perturbations
                run(f"--command generate_bounded --bounded_search_rho 0.1 {common}")


if __name__ == "__main__":
    main()
