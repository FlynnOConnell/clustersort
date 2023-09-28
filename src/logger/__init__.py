"""
======
Logger
======
Initialization of the module logger.

.. currentmodule:: logging

.. autosummary::
    :toctree: generated/

Colorized output, file and stream handlers are configured in :func:`configure_logger`.

"""
from logger.logger_config import *

__all__ = ["configure_logger", "ColoredFormatter", "ESTFormatter"]
