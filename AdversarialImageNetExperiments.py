import shlex
import subprocess
import sys
from pathlib import Path

# scripts and relative paths below are relative to the repository root
REPO_ROOT = Path(__file__).resolve().parent

num_images = 600
noise_epsilons = [0.5, 1.0]
# set to True to also search for latent adversarial perturbations
run_latent_search = False


def run(args: str) -> None:
    subprocess.run([sys.executable, "AdversarialImageNet.py", *shlex.split(args)], cwd=REPO_ROOT)


run("--command measure_accuracy --num_images 50000")
# run(f"--command generate_conventional_from_validation --num_images {num_images}")
# run(f"--command generate_conventional_from_gan --num_images {num_images}")

if run_latent_search:
    for noise_epsilon in noise_epsilons:
        run(
            f"--command generate_bounded --num_images {num_images} --noise_epsilon {noise_epsilon} "
            "--bounded_search_rho 0.1"
        )
        run(f"--command generate_minimum --num_images {num_images} --noise_epsilon {noise_epsilon}")
