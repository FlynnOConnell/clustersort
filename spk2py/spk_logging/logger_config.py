from __future__ import annotations

import logging
from pathlib import Path
import pytz
from datetime import datetime
from logging.handlers import RotatingFileHandler

class ESTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, pytz.timezone('US/Eastern'))
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime('%Y-%m-%d %H:%M:%S')

def configure_logger(name, log_file: Path | str, level=logging.INFO):
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a file handler with rotation
    file_handler = RotatingFileHandler(log_file, mode='a', maxBytes=int(1e6))
    file_handler.setFormatter(ESTFormatter('%(filename)s - %(asctime)s - %(levelname)s - %(message)s', datefmt='%m-%d-%Y %I:%M:%S %p'))
    file_handler.setLevel(level)

    # Create a stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ESTFormatter('%(filename)s - %(asctime)s - %(levelname)s - %(message)s', datefmt='%m-%d-%Y %I:%M:%S %p'))
    stream_handler.setLevel(level)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger