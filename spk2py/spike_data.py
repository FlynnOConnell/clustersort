from __future__ import annotations

import logging
from collections import namedtuple
from datetime import datetime
from math import floor
from pathlib import Path

import numpy as np
import pytz
from sonpy import lib as sp

from cluster import extract_waveforms, filter_signal

# Define a custom logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(filename)s - %(asctime)s - %(message)s',
    datefmt='%m-%d-%Y %I:%M:%S %p',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Set timezone to EST
class ESTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, pytz.timezone('US/Eastern'))
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime('%Y-%m-%d %H:%M:%S')

for handler in logger.handlers:
    handler.setFormatter(ESTFormatter('%(filename)s - %(asctime)s - %(message)s', datefmt='%m-%d-%Y %I:%M:%S %p'))

class SpikeData:
    UnitData = namedtuple("UnitData", ["slices", "times"])
    Segment = namedtuple("Segment", ["segment_number", "data"])
    """
    Container class for Spike2 data.

    Can be used as:
    - A dictionary, where the keys are the channel names and the values are the waveforms (LFP + Unit).
    - A list, where the elements are the channel names.
    - A boolean, where True means the file is empty and False means it is not.
    - A string, where the string is the filename stem.
    """
    def __init__(self, filepath: Path | str , exclude: tuple = ("Respirat", "Sniff", "RefBrain"),):
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
        """
        self._bandpass_low = 300
        self._bandpass_high = 3000
        self.errors = {}
        self.exclude = exclude
        self.empty = False
        self.filename = Path(filepath)
        self.sonfile = sp.SonFile(str(self.filename), True)
        self.bitrate = 32 if self.sonfile.is32file() else 64
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

    def filter_extract(
        self, waveforms, sampling_rate: int | float, overlap=0.5
    ):
        # Ensure the Nyquist-Shannon sampling theorem is satisfied
        if self.time_base > 1 / (2 * self.bandpass_high):
            raise ValueError(
                "Sampling rate is too low for the given bandpass filter frequencies."
            )
    
        segment_points = len(waveforms)
        overlap_points = int(overlap * segment_points)
    
        all_slices = []
        all_spike_times = []
    
        # Initialize start and end points for segmentation
        start, end = 0, segment_points
    
        while end <= len(waveforms):
            segment = waveforms[start:end]
            filtered_segment = filter_signal(
                segment, (self.bandpass_low, self.bandpass_high), sampling_rate
            )

            slices, spike_times = extract_waveforms(
                filtered_segment,
                sampling_rate,
            )
            spike_times = [time + start for time in spike_times]
    
            all_slices.extend(slices)
            all_spike_times.extend(spike_times)
    
            # Move the start and end for the next segment
            start = end - overlap_points
            end = start + segment_points
    
        return np.array(all_slices), all_spike_times
    
    
    def get_adc_channels(self, segment_duration: int = 300):
        segment_ticks = int(segment_duration / self.time_base)

        # Initialize master dictionaries
        self.lfp = {}
        self.unit = {}

        for idx in range(self.max_channels):
            title = self.sonfile.GetChannelTitle(idx)
            if self.sonfile.ChannelType(idx) == sp.DataType.Adc and title not in self.exclude and "LFP" not in title:

                sampling_rate = np.round(1 / (self.sonfile.ChannelDivide(idx) * self.time_base), 2)
                total_ticks = self.sonfile.ChannelMaxTime(idx)
                num_segments = int(total_ticks / segment_ticks)

                # Initialize list to hold all segments for this channel
                segments = []

                # Chunk the channel into segments
                for segment_num in range(num_segments):
                    tFrom = int(segment_num * segment_ticks)
                    tUpto = int(tFrom + segment_ticks)
                    tUpto = min(tUpto, total_ticks)  # Ensure we don't exceed the channel's available ticks

                    # Extract and filter waveforms for this chunk
                    waveforms = self.sonfile.ReadFloats(idx, int(2e9), tFrom, tUpto)
                    slices, spike_times = self.filter_extract(waveforms, sampling_rate)

                    # Create a Segment instance and populate its data
                    # There's an argument to make this a dictionary instead,
                    # but a namedtuple is more readable and has less overhead
                    segment = self.Segment(segment_number=segment_num, data=self.UnitData(slices=slices, times=spike_times))
                    segments.append(segment)
                self.unit[title] = segments

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
    def recording_length(self):
        """The total recording length, in seconds."""
        return self.max_time * self.time_base

    @property
    def bandpass_low(self):
        """The lower bound of the bandpass filter."""
        return self._bandpass_low

    @bandpass_low.setter
    def bandpass_low(self, value):
        """Set the lower bound of the bandpass filter."""
        self._bandpass_low = value

    @property
    def bandpass_high(self):
        """The upper bound of the bandpass filter."""
        return self._bandpass_high

    @bandpass_high.setter
    def bandpass_high(self, value):
        """Set the upper bound of the bandpass filter."""
        self._bandpass_high = value

if __name__ == "__main__":
    path = Path().home() / "data"
    files = [f for f in path.glob("*")]
    file = files[0]
    data = SpikeData(file, ("Respirat", "RefBrain", "Sniff"),)
    x = 5
