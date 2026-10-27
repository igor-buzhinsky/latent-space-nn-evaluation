"""Logging of text and figures."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt


class LogUtil:
    """
    Performs logging of text and images to a new timestamped directory. Logs are also printed to stdout. Figures are
    only shown in notebooks.
    """

    def _timestamp() -> str:
        return str(int(round(time.time() * 1000)))

    _dirname = Path("logdir_" + _timestamp())
    _created = False

    @staticmethod
    def set_custom_dirname(dirname: str | Path) -> None:
        """Sets the custom name for a logging directory. The previous contents of the directory will be deleted!"""
        LogUtil._dirname = Path(dirname)
        LogUtil._dirname.mkdir(parents=True, exist_ok=True)
        LogUtil._created = True
        for path in LogUtil._dirname.iterdir():
            path.unlink()

    @staticmethod
    def ensure_dir_existence() -> None:
        """Creates the directory for logging if it hasn't been created yet."""
        if not LogUtil._created:
            LogUtil._created = True
            LogUtil._dirname.mkdir(parents=True, exist_ok=True)

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
        with open(LogUtil._dirname / "log.txt", "a+", encoding="utf-8") as f:
            f.write(f"[time_ms={round(time.time() * 1000)}] {msg}\n")

    @staticmethod
    def savefig(prefix: str, pdf: bool = False) -> None:
        """
        Saves a figure to the logging directory. The filename is generated based on the current timestamp and the
        provided prefix.

        :param prefix: To be included in the filename.
        :param pdf: If `True`, save as PDF. If `False`, save as PNG.
        """
        LogUtil.ensure_dir_existence()
        fname = LogUtil._dirname / f"fig_{prefix}_{LogUtil._timestamp()}{'.pdf' if pdf else '.png'}"
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
        """Switches matplotlib backend to `'pdf'`."""
        matplotlib.use("pdf", force=True)
