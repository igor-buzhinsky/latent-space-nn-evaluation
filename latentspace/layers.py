"""
Simple PyTorch layers.
"""

from collections.abc import Callable

import torch


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
