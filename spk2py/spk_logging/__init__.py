"""
======
Logger
======
Initialization of the module logger.

.. currentmodule:: spk2py.spk_logging

.. autosummary::
    :toctree: generated/

Colorized output, file and stream handlers are configured in :func:`configure_logger`.

"""
from spk2py.spk_logging.logger_config import *

__all__ = ["configure_logger", "ColoredFormatter", "ESTFormatter"]