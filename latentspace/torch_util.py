"""PyTorch helpers: device handling, tensors and model parameters."""

import random
import time
from typing import TypeVar, cast

import numpy as np
import torch

# whether CUDA is used (if it is available, the first GPU is used)
USING_CUDA = torch.cuda.is_available()
# device to place tensors and models on
DEVICE = "cuda:0" if USING_CUDA else "cpu"

TensorOrModule = TypeVar("TensorOrModule", bound=torch.Tensor | torch.nn.Module)


def to_device(x: TensorOrModule) -> TensorOrModule:
    """
    Returns the tensor/module on GPU if there is at least 1 GPU, otherwise just returns the tensor/module.

    :param x: A PyTorch tensor or module.
    :return: `x` on GPU if there is at least 1 GPU, otherwise just `x`.
    """
    return cast(TensorOrModule, x.cuda()) if USING_CUDA else x


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor to a NumPy array.

    :param x: Input tensor (possibly on GPU or requiring gradients).
    :return: NumPy array with the data of `x`.
    """
    return x.detach().cpu().numpy()


def optimizable_clone(x: torch.Tensor) -> torch.Tensor:
    """
    Clones a PyTorch tensor and makes it suitable for optimization.

    :param x: Input tensor.
    :return: `x` with enabled gradients.
    """
    return to_device(x.clone().detach()).requires_grad_(True)


@torch.no_grad()
def init_weights(m: torch.nn.Module) -> None:
    """
    Randomly initializes the weights of a convolutional or linear layer. To be used with `torch.nn.Module.apply`.

    :param m: PyTorch module.
    """
    if isinstance(m, torch.nn.Conv2d):
        m.weight.normal_(0, 0.1)
        if m.bias is not None:
            m.bias.normal_(0, 0.1)
    elif isinstance(m, torch.nn.Linear):
        fan_in: int = m.weight.size(0)
        # heuristic
        std = (2 / fan_in) ** 0.5 / 20
        m.weight.uniform_(-std, std)
        if m.bias is not None:
            m.bias.uniform_(0, 0)


def set_param_requires_grad(m: torch.nn.Module, value: bool) -> None:
    """
    Sets `requires_grad_(value)` for all parameters of the module.

    :param m: PyTorch module.
    :param value: Value to set.
    """
    for p in m.parameters():
        p.requires_grad_(value)


def number_of_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Number of trainable parameters in a PyTorch module, including nested modules.

    :param model: PyTorch module.
    :return: Number of trainable parameters in `model`.
    """
    return int(sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad]))


def set_random_seed(seed: int | None = None) -> None:
    """
    Sets the random seed of `random`, NumPy and PyTorch.

    :param seed: Seed value. If `None`, it is replaced with the current timestamp.
    """
    if seed is None:
        seed = int(time.time())
    else:
        assert seed >= 0
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 2)


def normalize_latent(x: torch.Tensor) -> torch.Tensor:
    """
    Divides each latent vector of a batch by its scaled Euclidean norm.

    :param x: Batch of latent vectors.
    :return: Normalized batch.
    """
    return x * (np.sqrt(x.shape[1]) / torch.norm(x, dim=1, keepdim=True))
