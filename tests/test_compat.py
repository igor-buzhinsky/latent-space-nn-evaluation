import importlib.util
import sys
import unittest

import torch.hub

from latentspace.compat import patch_torchvision_for_robustness


class PatchTorchvisionForRobustnessTest(unittest.TestCase):
    def test_provides_load_state_dict_from_url(self) -> None:
        patch_torchvision_for_robustness()
        from torchvision.models.utils import load_state_dict_from_url  # pyrefly: ignore[missing-import]

        self.assertIs(load_state_dict_from_url, torch.hub.load_state_dict_from_url)

    def test_can_be_called_repeatedly(self) -> None:
        patch_torchvision_for_robustness()
        module = sys.modules["torchvision.models.utils"]
        patch_torchvision_for_robustness()
        self.assertIs(sys.modules["torchvision.models.utils"], module)

    @unittest.skipIf(importlib.util.find_spec("robustness") is None, "robustness is not installed")
    def test_robustness_can_be_imported(self) -> None:
        patch_torchvision_for_robustness()
        from robustness import datasets, model_utils  # noqa: F401


if __name__ == "__main__":
    unittest.main()
