import shlex
import subprocess
import sys

no_images = 600
noise_epsilons = [0.5, 1.0]
# set to True to also search for latent adversarial perturbations
run_latent_search = False


def run(args: str):
    subprocess.run([sys.executable, "AdversarialImageNet.py", *shlex.split(args)])


run("--command measure_accuracy --no_images 50000")
# run(f"--command generate_conventional_from_validation --no_images {no_images}")
# run(f"--command generate_conventional_from_gan --no_images {no_images}")

if run_latent_search:
    for noise_epsilon in noise_epsilons:
        run(
            f"--command generate_bounded --no_images {no_images} --noise_epsilon {noise_epsilon} "
            "--bounded_search_rho 0.1"
        )
        run(f"--command generate_minimum --no_images {no_images} --noise_epsilon {noise_epsilon}")
