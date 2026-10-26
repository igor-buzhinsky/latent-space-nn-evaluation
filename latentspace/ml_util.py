import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from latentspace.torch_util import normalize_latent, to_device

# An enumeration of all supported datasets.
DatasetInfo = Enum("DatasetInfo", "MNIST CelebA128Gender LSUN128 ImageNet")

### PyTorch utils ###


class Lambda(torch.nn.Module):
    def __init__(self, forward: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.lambda_forward = forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambda_forward(x)


class NopLayer(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Flatten(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], -1)


class Upsample(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(x, size=None, scale_factor=2, mode="bilinear", align_corners=False)


class Resize(torch.nn.Module):
    def __init__(self, side: int) -> None:
        super().__init__()
        self.side = side

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(x, size=(self.side, self.side), mode="bicubic", align_corners=False)


class Adversary(ABC):
    """
    Base class for adversaries. Adversaries can perturb vectors given the gradient pointing to the direction
    of making the prediction worse.
    """

    @abstractmethod
    def perturb(
        self, initial_vector: torch.Tensor, get_gradient: Callable[[torch.Tensor], tuple[torch.Tensor, float]]
    ) -> torch.Tensor:
        """
        Perturb the given vector.

        :param initial_vector: Initial vector. If this is the original image representation, it must be flattened
            prior to the call (more precisely, it must be of size [1, the_rest]).
        :param get_gradient: A `get_gradient` function. It accepts the current vector and returns a tuple
            (gradient pointing to the direction of the adversarial attack, the corresponding loss function value).
        :return: The perturbed vector of the same size as `initial_vector`.
        """


class NopAdversary(Adversary):
    """
    Dummy adversary that acts like an identity function.
    """

    def perturb(
        self, initial_vector: torch.Tensor, get_gradient: Callable[[torch.Tensor], tuple[torch.Tensor, float]]
    ) -> torch.Tensor:
        return initial_vector


class PGDAdversary(Adversary):
    """
    Performs Projected Gradient Descent (PGD), or, more precisely, PG ascent according to the provided gradient.
    """

    def __init__(
        self,
        rho: float = 0.1,
        steps: int = 25,
        step_size: float = 0.1,
        random_start: bool = True,
        stop_loss: float = 0,
        verbose: int = 1,
        norm: str = "scaled_l_2",
        n_repeat: int = 1,
        repeat_mode: str | None = None,
        unit_sphere_normalization: bool = False,
    ) -> None:
        """
        Constructs `PGDAdversary`.

        :param rho: Bound on the perturbation norm (positive number).
        :param steps: Number of steps to perform in each run. Fewer steps can be done if `stop_loss` is reached.
        :param step_size: Step size. Each step will be of magnitude `rho * step_size`.
        :param random_start: If `True`, start the search in a vector with a uniformly random radius within the
            rho-ball. Otherwise, start in the center of the rho-ball.
        :param stop_loss: The search will stop when this value of the "loss" function is exceeded.
        :param verbose: 0 (silent), 1 (regular), 2 (verbose).
        :param norm: One of `'scaled_l_2'` (default), `'l_2'` or `'l_inf'`.
        :param n_repeat: Number of times to run PGD.
        :param repeat_mode: `'any'` or `'min'`. In mode `'any'`, `n_repeat` runs are identical, and any run that
            reaches `stop_loss` will prevent subsequent runs. In mode `'min'`, all runs will be performed, and if a
            run finds a smaller perturbation according to `norm`, it will tighten `rho` on the next run.
        :param unit_sphere_normalization: Search perturbations on the unit sphere (according to the scaled L2 norm)
            instead of the entire latent space.
        """
        super().__init__()
        self.rho = rho
        self.steps = steps
        self.step_size = step_size
        self.random_start = random_start
        self.stop_loss = stop_loss
        self.verbose = verbose
        # checks on norms
        assert norm in ["scaled_l_2", "l_2", "l_inf"], "norm must be either 'scaled_l_2', 'l_2' or 'l_inf'"
        self.scale_norm = norm == "scaled_l_2"
        self.inf_norm = norm == "l_inf"
        # checks on repeated runs
        assert n_repeat >= 1, "n_repeat must be positive"
        assert not (n_repeat > 1 and repeat_mode is None), "if n_repeat > 1, repeat_mode must be set"
        assert repeat_mode in [None, "any", "min"], "if repeat_mode is set, it must be either 'any' or 'min'"
        self.n_repeat = n_repeat
        self.shrinking_repeats = repeat_mode == "min"
        # optional unit sphere normalization
        self.unit_sphere_normalization = unit_sphere_normalization
        assert (
            not unit_sphere_normalization or norm == "scaled_l_2"
        ), "unit_sphere_normalization is only compatible with scaled_l_2 norm"

    def _norm(self, x: torch.Tensor) -> float:
        """
        (Possibly scaled) norm of `x`.
        """
        return x.norm(np.inf if self.inf_norm else 2).item() / (np.sqrt(x.numel()) if self.scale_norm else 1)

    def _normalize_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the vector of gradients.
        In the L2 space, this is done by dividing the vector by its norm.
        In the L-inf space, this is done by taking the sign of the gradient.
        """
        return x.sign() if self.inf_norm else (x / self._norm(x))

    def _project(self, x: torch.Tensor, rho: float) -> torch.Tensor:
        """
        Projects the vector onto the rho-ball.
        In the L2 space, this is done by scaling the vector.
        In the L-inf space, this is done by clamping all components independently.
        """
        return x.clamp(-rho, rho) if self.inf_norm else (x / self._norm(x) * rho)

    def perturb(
        self, initial_vector: torch.Tensor, get_gradient: Callable[[torch.Tensor], tuple[torch.Tensor, float]]
    ) -> torch.Tensor:
        best_perturbation = None
        best_perturbation_norm = np.inf
        # rho may potentially shrink with repeat_mode == "min":
        rho = self.rho
        random_start = self.random_start
        for run_n in range(self.n_repeat):
            x1 = initial_vector * 1
            perturbation = x1 * 0

            if random_start:
                # random vector within the rho-ball
                if self.inf_norm:
                    # uniform
                    perturbation = (torch.rand(1, x1.numel()) - 0.5) * 2 * rho
                    # possibly reduce radius to encourage search of vectors with smaller norms
                    perturbation *= np.random.rand()
                else:
                    # uniform radius, random direction
                    # note that this distribution is not uniform in terms of R^n!
                    perturbation = torch.randn(1, x1.numel())
                    if rho > 0:
                        perturbation /= self._norm(perturbation) / rho
                    else:
                        perturbation *= 0
                    perturbation *= np.random.rand()
                perturbation = to_device(perturbation)

            if self.verbose > 0:
                print(f">> #run = {run_n}, ║x1║ = {self._norm(x1):.5f}, ρ = {rho:.5f}")

            found = False
            for i in range(self.steps):
                perturbed_vector = x1 + perturbation
                perturbed_vector, perturbation = self._recompute_with_unit_sphere_normalization(
                    perturbed_vector, perturbation
                )
                classification_gradient, classification_loss = get_gradient(perturbed_vector)
                if self.verbose > 0:
                    if classification_loss > self.stop_loss or i == self.steps - 1 or i % 5 == 0 and self.verbose > 1:
                        print(
                            f"step {i:3d}: objective = {-classification_loss:+.7f}, "
                            f"║Δx║ = {self._norm(perturbation):.5f}, ║x║ = {self._norm(perturbed_vector):.5f}"
                        )
                if classification_loss > self.stop_loss:
                    found = True
                    break
                # learning step
                perturbation_step = rho * self.step_size * self._normalize_gradient(classification_gradient)
                perturbation_step = self._adjust_step_for_unit_sphere(perturbation_step, x1 + perturbation)
                perturbation += perturbation_step
                # projecting on rho-ball around x1
                if self._norm(perturbation) > rho:
                    perturbation = self._project(perturbation, rho)

            # end of run
            if found:
                if self.shrinking_repeats:
                    if self._norm(perturbation) < best_perturbation_norm:
                        best_perturbation_norm = self._norm(perturbation)
                        best_perturbation = perturbation
                        rho = best_perturbation_norm
                else:  # regular repeats
                    # return immediately
                    return self._optional_normalize(x1 + perturbation)
            if best_perturbation is None:
                best_perturbation = perturbation
            if self.shrinking_repeats and run_n == self.n_repeat - 1:
                # heuristic: the last run is always from the center
                random_start = False
        assert best_perturbation is not None, "n_repeat must be positive"
        return self._optional_normalize(initial_vector + best_perturbation)

    ### projections on the unit sphere: ###

    def _optional_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optional unit sphere normalization.

        :param x: Vector of shape 1*dim to normalize.
        :return: Optionally normalized `x`.
        """
        return normalize_latent(x) if self.unit_sphere_normalization else x

    def _recompute_with_unit_sphere_normalization(
        self, perturbed_vector: torch.Tensor, perturbation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        If unit sphere normalization is enabled, the perturbed vector is projected on the unit sphere,
        and the perturbation vector is recomputed accordingly. Otherwise, returns the inputs unmodified.

        :param perturbed_vector: Perturbed vector (initial vector + perturbation).
        :param perturbation: Perturbation vector.
        :return: Possibly recomputed `(perturbed_vector, perturbation)`.
        """
        effective_perturbed_vector = self._optional_normalize(perturbed_vector)
        return effective_perturbed_vector, perturbation + effective_perturbed_vector - perturbed_vector

    def _adjust_step_for_unit_sphere(
        self, perturbation_step: torch.Tensor, previous_perturbed_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        If unit sphere normalization is enabled, multiplies `perturbation_step` by a coefficient that approximately
        compensates for the reduction of the learning step due to projection onto the unit sphere.

        :param perturbation_step: Unmodified perturbation step.
        :param previous_perturbed_vector: Previous perturbed vector.
        :return: Altered `perturbation_step`.
        """
        new_perturbed_vector = self._optional_normalize(previous_perturbed_vector + perturbation_step)
        effective_perturbation_step = new_perturbed_vector - previous_perturbed_vector
        coef = perturbation_step.norm() / effective_perturbation_step.norm()
        return perturbation_step * coef


class EpsDTransformer:
    """
    Converter between noise magnitude (epsilon) and decay factor (d).
    """

    def eps_to_d(self, eps: float) -> float:
        """
        Converts epsilon to d.

        :param eps: Noise magnitude.
        :return: Decay factor.
        """
        return 1 - 1 / np.sqrt(1 + eps**2)

    def d_to_eps(self, d: float) -> float:
        """
        Converts d to epsilon.

        :param d: Decay factor.
        :return: Noise magnitude.
        """
        return np.sqrt((1 / (1 - d)) ** 2 - 1)


class TauRhoTransformer:
    """
    Converter between perturbation likelihood (tau) and norm bound (rho).
    """

    def __init__(self, n_l: int, eps: float) -> None:
        self.n_l = n_l
        self.c1 = n_l * np.log(np.sqrt((1 + eps**2) / (2 * np.pi * eps**2)))
        self.c2 = (1 + eps**2) / (2 * eps**2) * np.sqrt(n_l)

    def rho_to_logtau(self, rho: float) -> float:
        """
        Converts rho to the logarithm of tau.
        """
        return self.c1 - (rho**2) * self.n_l * self.c2

    def logtau_to_rho(self, tau: float) -> float:
        """
        Converts the logarithm of tau to rho.
        """
        return np.sqrt((self.c1 - tau) / self.c2 / self.n_l)


class LogUtil:
    """
    Performs logging of text and images to a new timestamped directory.
    Logs are also printed to stdout. Figures are only shown in notebooks.
    """

    def _timestamp() -> str:
        return str(int(round(time.time() * 1000)))

    _dirname = "logdir_" + _timestamp()
    _created = False

    @staticmethod
    def set_custom_dirname(dirname: str) -> None:
        """
        Sets the custom name for a logging directory.
        The previous contents of the directory will be deleted!
        """
        LogUtil._dirname = dirname
        LogUtil.ensure_dir_existence()
        for filename in os.listdir(dirname):
            os.remove(os.path.join(dirname, filename))

    @staticmethod
    def ensure_dir_existence() -> None:
        """
        Creates the directory for logging if it hasn't been created yet.
        """
        if not LogUtil._created:
            LogUtil._created = True
            os.makedirs(LogUtil._dirname, exist_ok=True)

    @staticmethod
    def info(msg: Any, regular_print: bool = True) -> None:
        """
        Logs a message to `<log directory>/log.txt` and optionally prints it.

        :param msg: Message to log and, optionally, print to console.
        :param regular_print: Whether to print `msg` to console.
        """
        LogUtil.ensure_dir_existence()
        if regular_print:
            print(msg)
        with open(os.path.join(LogUtil._dirname, "log.txt"), "a+", encoding="utf-8") as f:
            f.write(f"[time_ms={round(time.time() * 1000)}] {msg}\n")

    @staticmethod
    def savefig(prefix: str, pdf: bool = False) -> None:
        """
        Saves a figure to the logging directory. The filename is generated based on the current timestamp
        and the provided prefix.

        :param prefix: To be included in the filename.
        :param pdf: If `True`, save as PDF. If `False`, save as PNG.
        """
        LogUtil.ensure_dir_existence()
        fname = os.path.join(
            LogUtil._dirname, "fig_" + prefix + "_" + LogUtil._timestamp() + (".pdf" if pdf else ".png")
        )
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        LogUtil.info(f"[produced a figure: {fname}]", False)

    @staticmethod
    def metacall(fn: Callable, fn_name: str, *args: Any, **kwargs: Any) -> None:
        """
        Makes a function call and logs it with concrete argument values.

        :param fn: Function to call.
        :param fn_name: The name of the function to be put to the log.
        :param args: `*args` to be passed to `fn`.
        :param kwargs: `**kwargs` to be passed to `fn`.
        """
        arg_assignments = [repr(arg) for arg in args]
        kwarg_assignments = [f"{kwarg_name}={repr(kwarg_value)}" for kwarg_name, kwarg_value in kwargs.items()]
        LogUtil.info(f"{fn_name}(" + ", ".join(arg_assignments + kwarg_assignments) + ")", False)
        fn(*args, **kwargs)

    @staticmethod
    def to_pdf() -> None:
        """
        Switches matplotlib backend to `'pdf'`.
        """
        matplotlib.use("pdf", force=True)
