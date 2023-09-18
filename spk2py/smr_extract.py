from __future__ import annotations

from math import floor
from pathlib import Path
import numpy as np
from spk2py.spk_io import h5
import logging
from sonpy import lib as sp
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmrExtract:
    def __init__(self, filepath):
        self.filename = Path(filepath)
        self.sonfile = sp.SonFile(str(self.filename), True)


    def get_adc_channels(self):
        for x in range(self.max_channels):
            print(( self.sonfile.GetChannelTitle(x), self.sonfile.ChannelType(x)))


    @property
    def time_base(self):
        return self.sonfile.GetTimeBase()

    @property
    def max_time(self):
        return self.sonfile.MaxTime()

    @property
    def max_channels(self):
        return self.sonfile.MaxChannels()

    @property
    def version(self):
        return self.sonfile.GetVersion()

    @property
    def bitrate(self):
        return 32 if self.sonfile.is32file() else 64






