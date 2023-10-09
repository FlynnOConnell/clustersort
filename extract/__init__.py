"""
===============
``spk2extract``
===============

*spk2extract. A spike 2 data extraction utility for extracellular recordings.*

Documentation Guide
-------------------

I recommend exploring the docstrings using
`IPython <https://ipython.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.

"""
import platform

from platformdirs import *

import gui
from .defaults import defaults  # noqa (API import)
from .extraction import SpikeData, UnitData  # noqa (API import)
from .gui import *  # noqa (API import)
from .spk_io import *  # noqa (API import)
from .util import *  # noqa (API import)
from .version import version as __version__  # noqa (API import)

def is_m1_mac():
    if platform.system() == 'Darwin':  # Darwin indicates macOS
        uname_info = platform.uname()
        if uname_info.machine == 'arm64':
            return True
    return False

version = __version__
__name__ = "extract"  # This module is distinct, and should be imported as such
__author__ = "Flynn OConnell"

# Platform-dependent directories
cache_dir = user_cache_dir(__name__, __author__)  # Cache, temp files
config_dir = user_config_dir(__name__, __author__)  # Config, parameters and options
log_dir = user_log_dir(__name__, __author__)  # Logs, .log files primarily

# Documentation inclusions
__all__ = [
    "SpikeData",
    "UnitData",
    "spk_io",
    "extract_waveforms",
    "dejitter",
    "filter_signal",
    "version",
    "gui",
    "cache_dir",
    "config_dir",
    "log_dir",
]
