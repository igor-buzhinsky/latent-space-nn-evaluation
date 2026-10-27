"""Locations of data and models in the repository, independent of the current directory."""

from pathlib import Path

# root of the repository (the parent of the latentspace package)
REPO_ROOT = Path(__file__).resolve().parents[1]
# datasets (except MNIST, which is downloaded to REPO_ROOT / "mnist")
DATA_DIR = REPO_ROOT / "data"
