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


def raise_open_file_limit(limit: int = 4096) -> None:
    """
    Raises the limit on the number of open files (this has effect only on POSIX systems). This may prevent errors in
    data loaders with `num_workers > 0`.

    :param limit: Desired limit. It is capped by the hard limit of the system.
    """
    if os.name == "posix":
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, limit), hard))


def configure_runtime(memory_limit_mb: int) -> None:
    """
    Sets memory limit in megabytes (this has effect only on Linux), raises the open file limit, and configures
    matplotlib fonts.
    """
    set_memory_limit(memory_limit_mb)
    raise_open_file_limit()
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
