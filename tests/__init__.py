"""
Tests of the latentspace package. Run them from the repository root (bundled models are loaded by relative paths):

    poetry run python -m unittest discover -b tests

(`-b` hides the output of passing tests.)
"""

import atexit
import shutil
import tempfile

import matplotlib
import matplotlib.pyplot as plt

from latentspace.logging_util import LogUtil

# figures are only saved, never shown
matplotlib.use("Agg")
plt.show = lambda *args, **kwargs: None

# write logs and figures produced during tests to a temporary directory instead of the repository root
_LOG_DIR = tempfile.mkdtemp(prefix="latentspace-test-logs-")
LogUtil.set_custom_dirname(_LOG_DIR)
atexit.register(shutil.rmtree, _LOG_DIR, ignore_errors=True)
