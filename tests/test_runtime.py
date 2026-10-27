import os
import subprocess
import sys
import unittest


def run_python(code: str) -> str:
    """Runs code in a separate interpreter, so that process-level settings do not affect the tests."""
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()


@unittest.skipUnless(os.name == "posix", "memory limits are only supported on POSIX systems")
class SetMemoryLimitTest(unittest.TestCase):
    def test_sets_data_segment_limit(self) -> None:
        output = run_python(
            "import resource\n"
            "from latentspace.runtime import set_memory_limit\n"
            "set_memory_limit(20000)\n"
            "print(resource.getrlimit(resource.RLIMIT_DATA)[0])"
        )
        self.assertEqual(int(output), 20000 * 1024 * 1024)


@unittest.skipUnless(os.name == "posix", "open file limits are only supported on POSIX systems")
class RaiseOpenFileLimitTest(unittest.TestCase):
    def test_raises_soft_limit_up_to_hard_limit(self) -> None:
        output = run_python(
            "import resource\n"
            "from latentspace.runtime import raise_open_file_limit\n"
            "soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)\n"
            "resource.setrlimit(resource.RLIMIT_NOFILE, (min(256, soft), hard))\n"
            "raise_open_file_limit(1024)\n"
            "new_soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]\n"
            "print(new_soft == (1024 if hard == resource.RLIM_INFINITY else min(1024, hard)))"
        )
        self.assertEqual(output, "True")

    def test_does_not_lower_the_limit(self) -> None:
        output = run_python(
            "import resource\n"
            "from latentspace.runtime import raise_open_file_limit\n"
            "before = resource.getrlimit(resource.RLIMIT_NOFILE)[0]\n"
            "raise_open_file_limit(1)\n"
            "print(resource.getrlimit(resource.RLIMIT_NOFILE)[0] == before)"
        )
        self.assertEqual(output, "True")


class ConfigureRuntimeTest(unittest.TestCase):
    def test_configures_matplotlib_fonts(self) -> None:
        output = run_python(
            "import matplotlib.pyplot as plt\n"
            "from latentspace.runtime import configure_runtime\n"
            "configure_runtime(20000)\n"
            "print(plt.rcParams['font.family'], plt.rcParams['mathtext.fontset'])"
        )
        self.assertEqual(output, "['serif'] dejavuserif")


if __name__ == "__main__":
    unittest.main()
