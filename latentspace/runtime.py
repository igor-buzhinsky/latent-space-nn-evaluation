"""Process-level configuration of scripts."""

import os

import matplotlib.pyplot as plt

if os.name == "posix":
    import resource


def set_memory_limit(mb: int) -> None:
    """Sets memory limit in megabytes (this has effect only on Linux)."""
    if os.name == "posix":
        rsrc = resource.RLIMIT_DATA
        _, hard = resource.getrlimit(rsrc)
        resource.setrlimit(rsrc, (1024 * 1024 * mb, hard))


def configure_runtime(memory_limit_mb: int) -> None:
    """Sets memory limit in megabytes (this has effect only on Linux). Configures matplotlib fonts."""
    set_memory_limit(memory_limit_mb)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
