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

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[33m',  # Orange
        'INFO': '\033[34m',   # Blue
        'WARNING': '\033[31m',  # Red
        'ERROR': '\033[41m',  # Red background
        'CRITICAL': '\033[45m'  # Magenta background
    }

    def format(self, record):
        log_message = super().format(record)
        return f"{self.COLORS.get(record.levelname, '')}{log_message}\033[0m"

def configure_logger(name, log_file: Path | str, level=logging.INFO):
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    logger.handlers = []

    # Create a file handler with rotation
    file_handler = RotatingFileHandler(log_file, mode='a', maxBytes=int(1e6))
    file_handler.setFormatter(ESTFormatter('%(filename)s - %(asctime)s - %(levelname)s - %(message)s', datefmt='%m-%d-%Y %I:%M:%S %p'))
    file_handler.setLevel(level)

    # Create a stream handler with color
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ColoredFormatter('%(filename)s - %(asctime)s - %(levelname)s - %(message)s', datefmt='%m-%d-%Y %I:%M:%S %p'))
    stream_handler.setLevel(level)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger