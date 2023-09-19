from __future__ import annotations

import logging
from math import floor
from pathlib import Path

from sonpy import lib as sp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpikeData:
    """
    Container class for Spike2 data.

    Can be used as:
    - A dictionary, where the keys are the channel names and the values are the waveforms (LFP + Unit).
    - A list, where the elements are the channel names.
    - A boolean, where True means the file is empty and False means it is not.
    - A string, where the string is the filename stem.
    - A Path object, for filename.stem, name, parent, suffix, absolute, and exists properties.
    """
    def __init__(self, filepath: Path | str, exclude: tuple = (),):
        """
        Class for reading and storing data from a Spike2 file.

        Parameters:
        -----------
        filepath : Path | str
            The full path to the Spike2 file, including filename + extension.
        exclude : tuple, optional
            A tuple of channel names to exclude from the data. Default is empty tuple.

        Attributes:
        -----------
        exclude : tuple
            A tuple of channel names to exclude from the data.
        empty : bool
            Whether the file is empty or not.
        filename : Path
            The full path to the Spike2 file, including filename + extension.
        sonfile : SonFile
            The SonFile object from the sonpy library.
        lfp : dict
            A dictionary of LFP channels, where the keys are the channel names and the values are the waveforms.
        unit : dict
            A dictionary of unit channels, where the keys are the channel names and the values are the waveforms.
        preinfusion : bool
            Whether the file is a pre-infusion file or not.
        postinfusion : bool
            Whether the file is a post-infusion file or not.
        time_base : float
            Everything in the file is quantified by the underlying clock tick (64-bit).
            All values in the file are stored, set and returned in ticks.
            You need to read this value to interpret times in seconds.
        max_time : float
            The last time-point in the array, in ticks.
        max_channels : int
            The number of channels in the file.
        bitrate : int
            Whether the file is 32bit (old) or 64bit (new).
        recording_length : float
            The total recording length, in seconds.
        stem : str
            The filename without the extension.
        suffix : str
            The filename extension.
        name : str
            The filename with extension.
        parent : Path
            The parent directory path
        absolute : Path
            The absolute path to the file.
        exists : bool
            Whether the file exists or not.
        """
        self.exclude = exclude
        self.empty = False
        self.filename = Path(filepath)
        self.sonfile = sp.SonFile(str(self.filename), True)
        self.lfp = {}
        self.unit = {}
        self.get_adc_channels()
        self._validate()

    def __repr__(self):
        return f"{self.filename.stem}"

    def __str__(self):
        """Allows us to use str(spike_data.SpikeData(file)) to get the filename stem."""
        return f"{self.filename.stem}"

    def __bool__(self):
        return self.empty

    def __getitem__(self, key):
        if key in self.lfp:
            return self.lfp[key]
        elif key in self.unit:
            return self.unit[key]
        else:
            raise KeyError(f"{key} not found in SpikeData object.")

    def __setitem__(self, key, value):
        if key in self.lfp:
            self.lfp[key] = value
        elif key in self.unit:
            self.unit[key] = value
        else:
            raise KeyError(f"{key} not found in SpikeData object.")

    def _validate(self):
        """General checks to make sure the data is valid"""

        # with empty waveform data, the size of each channel array is 1 for some reason
        # the check for > 20 is arbitrary, theoretically it could be a lot more
        for k, v in self.lfp.items():
            if v.size < 20:
                self.lfp[k] = {}
        for k, v in self.unit.items():
            if v.size < 20:
                self.unit[k] = {}
                self.empty = True

        if self.preinfusion and self.postinfusion:
            raise ValueError(
                f"File {self.filename.stem} contains both a pre- and post-infusion filename."
            )
        elif not self.preinfusion and not self.postinfusion:
            raise ValueError(
                f"File {self.filename.stem} does not contain a pre- or post-infusion filename."
            )

    @property
    def preinfusion(self):
        return "pre" in self.filename.stem

    @property
    def postinfusion(self):
        return "post" in self.filename.stem

    def get_adc_channels(self):
        """Fill dictionaries for LFP and unit channels.

        If there are gaps in the Spike2 file, this will not be read as zero or NaN etc., it will simply be passed over,
        and  we end up with fewer points than the channel divide would otherwise.
        """
        for idx in range(self.max_channels):
            title = self.sonfile.GetChannelTitle(idx)
            if self.sonfile.ChannelType(idx) == sp.DataType.Adc and title not in self.exclude:
                # this will always show "parameter filter is not used" because SonPy hides the actual parameter filter

                # we want all waveforms, so we want max_ticks to exceed the number of ticks in the file
                # if this is a 32bit file, the max number of ticks is 2e9 (technically 2.147e9, but close enough)
                # if this is a 64bit file, the max number of ticks is 1e12 (...1.844e19, close enough!)
                if self.bitrate == 32:
                    max_ticks = 2e9
                else:
                    max_ticks = 1e12
                waveforms = self.sonfile.ReadFloats(idx, max_ticks, 0)
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
    
    def channel_max_time(self, channel: int):
        """The last time-point in the array, in ticks."""
        return self.sonfile.ChannelMaxTime(channel)

    @property
    def time_base(self):
        """
        Everything in the file is quantified by the underlying clock tick (64-bit).
        All values in the file are stored, set and returned in ticks.
        You need to read this value to interpret times in seconds.

        Returns
        -------
        float
            The time base, in seconds.
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

    @property
    def stem(self):
        """The filename without the extension."""
        return self.filename.stem

    @property
    def suffix(self):
        """The filename extension."""
        return self.filename.suffix

    @property
    def name(self):
        """The filename with extension."""
        return self.filename.name

    @property
    def parent(self):
        """The parent directory path"""
        return self.filename.parent

    @property
    def absolute(self):
        return self.filename.absolute()

    @property
    def exists(self):
        return self.filename.exists()


if __name__ == "__main__":
    path = Path().home() / "data" / "smr"
    files = [f for f in path.glob("*")]
    file = files[0]
    data = SpikeData(file, ("Respirat", "RefBrain", "Sniff"),)
    x = 5
