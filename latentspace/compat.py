"""
Compatibility shims for third-party packages written for older library versions.
"""

import importlib.util
import sys
import types

import torch.hub


def patch_torchvision_for_robustness():
    """
    The robustness package (version 1.2.1) imports load_state_dict_from_url from torchvision.models.utils,
    which was removed in newer torchvision versions. This recreates that module, so robustness can be imported.
    Call this before importing robustness.
    """
    if importlib.util.find_spec("torchvision.models.utils") is None:
        module = types.ModuleType("torchvision.models.utils")
        module.load_state_dict_from_url = torch.hub.load_state_dict_from_url  # pyrefly: ignore[missing-attribute]
        sys.modules["torchvision.models.utils"] = module
