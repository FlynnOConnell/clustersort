from __future__ import annotations

import logging
from collections import namedtuple
from math import floor
from pathlib import Path

import h5py
import numpy as np
from sonpy import lib as sp

from cluster import extract_waveforms, filter_signal
from spk_logging.logger_config import configure_logger

logfile = Path().home() / "data" / "spike_data.log"
logger = configure_logger(__name__, logfile, level=logging.DEBUG)

Segment = namedtuple("Segment", ["segment_number", "data"])
UnitData = namedtuple("UnitData", ["slices", "times"])

def load_from_h5(filename):
    # Dictionary to hold the loaded data
    data_dict = {}
    with h5py.File(filename, "r") as f:
        # Load metadata
        metadata_grp = f["metadata"]
        data_dict["metadata"] = {
            attr: metadata_grp.attrs[attr] for attr in metadata_grp.attrs
        }

        # Load unit data
        unit_grp = f["unit"]
        data_dict["unit"] = {}

        for title in unit_grp.keys():
            channel_grp = unit_grp[title]
            segments = []

            for segment_name in channel_grp.keys():
                segment_grp = channel_grp[segment_name]

                slices = np.array(segment_grp["slices"])
                times = np.array(segment_grp["times"])

                segment = Segment(
                    segment_number=int(segment_name.split("_")[1]),
                    data=UnitData(slices=slices, times=times),
                )
                segments.append(segment)

            data_dict["unit"][title] = segments

    return data_dict

class SpikeData:
    UnitData = namedtuple("UnitData", ["spikes", "times"])
    """
    Container class for Spike2 data.

    Can be used as:
    - A dictionary, where the keys are the channel names and the values are the waveforms (LFP + Unit).
    - A list, where the elements are the channel names.
    - A boolean, where True means the file is empty and False means it is not.
    - A string, where the string is the filename stem.
    """
    def __init__(
        self,
        filepath: Path | str,
        exclude: tuple = ("Respirat", "Sniff", "RefBrain"),
    ):
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
        self.process_units()
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
        if self.preinfusion and self.postinfusion:
            raise ValueError(
                f"File {self.filename.stem} contains both a pre- and post-infusion filename."
            )
        elif not self.preinfusion and not self.postinfusion:
            raise ValueError(
                f"File {self.filename.stem} does not contain a pre- or post-infusion filename."
            )

    def save_to_h5(self, filename):
        with h5py.File(filename, "w") as f:
            logger.debug("Setting metadata...")
            # All metadata we may need later
            metadata_grp = f.create_group("metadata")
            metadata_grp.attrs["bandpass_low"] = self.bandpass_low
            metadata_grp.attrs["bandpass_high"] = self.bandpass_high
            metadata_grp.attrs["time_base"] = self.time_base
            metadata_grp.attrs["max_time"] = self.max_time
            metadata_grp.attrs["max_channels"] = self.max_channels
            metadata_grp.attrs["bitrate"] = self.bitrate
            metadata_grp.attrs["recording_length"] = self.recording_length
            metadata_grp.attrs["infusion"] = "pre" if self.preinfusion else "post"
            metadata_grp.attrs["filename"] = self.filename.stem
            metadata_grp.attrs["empty"] = self.empty
            metadata_grp.attrs["exclude"] = self.exclude
            logger.debug(f"Saved metadata to {filename}")

            # Create a group for unit data
            unit_grp = f.create_group("unit")

            for title, unit_data in self.unit.items():
                # Create a subgroup for each channel title
                channel_grp = unit_grp.create_group(title)

                # Save spikes and times as datasets within the channel group
                channel_grp.create_dataset("spikes", data=unit_data.spikes)
                channel_grp.create_dataset("times", data=unit_data.times)

        logger.debug(f"Saved data successfully to {filename}")

    def save_data(self, savepath, overwrite=False):
        """Save the data to an HDF5 file."""
        path = Path(savepath)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        savename = f"{savepath}/{self.filename.stem}.h5"
        logger.debug(f"Saving data to {savename}")
        if Path(savename).exists() and overwrite:
            logger.debug(f"Overwriting {savename}")
            self.save_to_h5(savename)
        elif Path(savename).exists() and not overwrite:
            logger.debug(f"{savename} already exists, skipping.")
            return None
        else:
            logger.debug(f"Filename is fresh - saving {savename}")
            self.save_to_h5(savename)

    @property
    def preinfusion(self):
        return "pre" in self.filename.stem

    @property
    def postinfusion(self):
        return "post" in self.filename.stem

    def process_units(self, segment_duration: int = 300):
        """
        Extracts unit data from the Spike2 file.

        Args:
        -----
            segment_duration (int): The duration of each segment, in seconds.

        Returns:
        --------
            None
        """
        logger.debug(f"Extracting ADC channels from {self.filename.stem}")
        segment_ticks = int(segment_duration / self.time_base)

        for idx in range(self.max_channels):
            all_spikes = []
            all_times = []
            logger.debug(f"Max channels: {self.max_channels}")
            title = self.sonfile.GetChannelTitle(idx)
            if (
                self.sonfile.ChannelType(idx) == sp.DataType.Adc
                and title not in self.exclude
                and "LFP" not in title
            ):
                logger.debug(f"Processing {title}")
                sampling_rate = np.round(
                    1 / (self.sonfile.ChannelDivide(idx) * self.time_base), 2
                )
                total_ticks = self.sonfile.ChannelMaxTime(idx)
                num_segments = int(total_ticks / segment_ticks)

                # Chunk the channel into segments
                for segment_num in range(num_segments):
                    logger.debug(f"Processing segment {segment_num} of {num_segments}")
                    seg_time_start = int(segment_num * segment_ticks)
                    seg_time_end = int(seg_time_start + segment_ticks)

                    # Ensure we don't exceed the channel's available ticks
                    seg_time_end = min(
                        seg_time_end, total_ticks
                    )

                    # Extract and filter waveforms for this chunk
                    waveforms = self.sonfile.ReadFloats(idx, int(2e9), seg_time_start, seg_time_end)

                    # Ensure the Nyquist-Shannon sampling theorem is satisfied
                    if self.time_base > 1 / (2 * self.bandpass_high):
                        raise ValueError(
                            "Sampling rate is too low for the given bandpass filter frequencies."
                        )

                    # Low/high bandpass filter
                    filtered_segment = filter_signal(
                        waveforms, (self.bandpass_low, self.bandpass_high), sampling_rate
                    )

                    # Extract spikes and times from the filtered segment
                    slices, spike_times = extract_waveforms(
                        filtered_segment,
                        sampling_rate,
                    )
                    spike_times = [time + (segment_num * segment_ticks * sampling_rate) for time in spike_times]

                    all_spikes.append(slices)
                    all_times.append(spike_times)

                # Flatten and concatenate all spikes and times into single 1D numpy arrays
                final_spikes = np.concatenate(all_spikes).flatten()
                final_times = np.concatenate(all_times).flatten()
                final_times = np.round(final_times / sampling_rate, 3)

                # Create a FinalUnitData namedtuple with the concatenated spikes and times
                final_unit_data = self.UnitData(spikes=final_spikes, times=final_times)

                # Store this namedtuple in the self.unit dictionary
                self.unit[title] = final_unit_data

    def get_channel_interval(self, channel: int):
        """
        Get the waveform sample interval, in clock ticks. Used by channels that sample
        equal interval waveforms and = the number of file clock ticks per second.
        """
        return self.sonfile.ChannelDivide(channel)

    def get_channel_period(self, channel: int):
        """Get the waveform sample period, in seconds."""
        return self.get_channel_interval(channel) / self.time_base

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
    path_test = Path().home() / "data"
    files = [f for f in path_test.glob("*.smr")]
    file = files[0]
    data = SpikeData(
        file,
        ("Respirat", "RefBrain", "Sniff"),
    )
    data.save_data(path_test, overwrite=True)
    x = 5
