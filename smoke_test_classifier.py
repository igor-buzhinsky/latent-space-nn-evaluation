"""Smoke test: evaluates bundled classifiers on a real dataset with ClassifierTraining.py, using few images."""

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
        "--prefixes",
        nargs="+",
        choices=PREFIXES,
        default=["plain"],
        help="Classifiers to check, by training mode (default: %(default)s).",
    )
    parser.add_argument(
        "--img_accuracy",
        type=int,
        default=1000,
        help="Number of images for accuracy evaluation (default: %(default)s; full evaluation uses 100000).",
    )
    parser.add_argument(
        "--img_robustness",
        type=int,
        default=6,
        help="Number of images for robustness evaluation (default: %(default)s; full evaluation uses 600).",
    )
    return parser


def run(args: str) -> None:
    print(f"*** ClassifierTraining.py {args}", flush=True)
    result = subprocess.run([sys.executable, "ClassifierTraining.py", *shlex.split(args)], cwd=REPO_ROOT)
    if result.returncode != 0:
        # the error has already been reported by the script; stop at the first failure
        sys.exit(result.returncode)


def main() -> None:
    args = build_parser().parse_args()
    for unit_type in args.unit_types:
        for dataset in args.datasets:
            for prefix in args.prefixes:
                run(
                    f"--dataset {dataset} --command evaluate "
                    f'--load_filename "classifiers_architecture{unit_type}/{prefix}_{dataset}.bin" '
                    f"--img_accuracy {args.img_accuracy} --img_robustness {args.img_robustness} "
                    f"--unit_type {unit_type}"
                )


if __name__ == "__main__":
    main()
