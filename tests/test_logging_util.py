import os
import tempfile
import unittest

import matplotlib.pyplot as plt

from latentspace.logging_util import LogUtil
from tests.helpers import read_log


class LogUtilTest(unittest.TestCase):
    def test_info_appends_to_log_file(self) -> None:
        LogUtil.info("first test message", regular_print=False)
        LogUtil.info(42, regular_print=False)
        log = read_log()
        self.assertIn("first test message\n", log)
        self.assertIn("] 42\n", log)

    def test_savefig_creates_file(self) -> None:
        plt.figure()
        plt.plot([0, 1], [1, 0])
        LogUtil.savefig("test_prefix")
        plt.close()
        self.assertTrue(
            any(f.startswith("fig_test_prefix_") and f.endswith(".png") for f in os.listdir(LogUtil._dirname))
        )

    def test_metacall_calls_function_and_logs_arguments(self) -> None:
        results = []
        LogUtil.metacall(lambda x, y: results.append(x + y), "add", 1, y=2)
        self.assertEqual(results, [3])
        self.assertIn("add(1, y=2)", read_log())

    def test_set_custom_dirname_clears_directory(self) -> None:
        previous = LogUtil._dirname
        with tempfile.TemporaryDirectory() as dirname:
            open(os.path.join(dirname, "old.txt"), "w").close()
            LogUtil.set_custom_dirname(dirname)
            try:
                self.assertEqual(os.listdir(dirname), [])
                LogUtil.info("new message", regular_print=False)
                self.assertIn("new message", read_log())
            finally:
                LogUtil.set_custom_dirname(previous)


if __name__ == "__main__":
    unittest.main()
