from __future__ import annotations

import logging
from math import floor
from pathlib import Path

from sonpy import lib as sp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpikeData:
    def __init__(self, filepath: Path | str, exclude: tuple = ()):
        """
        Class for reading and storing data from a Spike2 file.

        Parameters:
        -----------
        filepath : Path | str
            The path to the Spike2 file.
        exclude : tuple, optional
            A tuple of channel names to exclude from the data. Default is empty tuple.
        """
        self.exclude = exclude
        self.empty = False
        self.filename = Path(filepath)
        self.sonfile = sp.SonFile(str(self.filename), True)
        self.lfp = {}
        self.unit = {}
        self.get_adc_channels()

    def __repr__(self):
        return f"{self.filename.stem}"

    def get_adc_channels(self):
        for idx in range(self.max_channels):
            title = self.sonfile.GetChannelTitle(idx)
            if self.sonfile.ChannelType(idx) == sp.DataType.Adc and title not in self.exclude:
                waveforms = self.sonfile.ReadFloats(idx, self.num_ticks(idx), 0)
                if "LFP" in title:
                    self.lfp[title] = waveforms
                elif "U" in title:
                    self.unit[title] = waveforms

        if not self.lfp and not self.unit:
            self.empty = True

    def get_channel_interval(self, channel: int):
        """
        Get the waveform sample interval, in clock ticks. Used by channels that sample
        equal interval waveforms and = the number of file clock ticks per second.
        """
        return self.sonfile.ChannelDivide(channel)

    def get_channel_period(self, channel: int):
        """Get the waveform sample period, in seconds."""
        return self.get_channel_interval(channel) / self.time_base

    def get_waveform_time(self,):
        """ Create a numpy array of time values for the waveform. """
        # TODO: Implement this
        # time = np.arange(0, len(wavedata) * dPeriod, dPeriod)
        pass

    def num_ticks(self, channel: int):
        """The total number of clock ticks for this channel."""
        return floor(self.recording_length / self.get_channel_period(channel))

    @property
    def time_base(self):
        """
        Everything in the file is quantified by the underlying clock tick (64-bit).
        All values in the file are stored, set and returned in ticks.
        You need to read this value to interpret times in seconds.
        """
        return self.sonfile.GetTimeBase()

    @property
    def max_time(self):
        """The last time-point in the array, in ticks."""
        return self.sonfile.MaxTime()

    @property
    def max_channels(self):
        """The number of channels in the file."""
        return self.sonfile.MaxChannels()

    @property
    def bitrate(self):
        """Whether the file is 32bit (old) or 64bit (new)."""
        return 32 if self.sonfile.is32file() else 64

    @property
    def recording_length(self):
        """The total recording length, in seconds."""
        return self.max_time * self.time_base


if __name__ == "__main__":
    path = Path().home() / "data" / "smr"
    files = [f for f in path.glob("*")]
    data = SpikeData(files[0], ("Respirat", "RefBrain", "Sniff"))
    x = 5
