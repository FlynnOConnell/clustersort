from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from spike_data import SpikeData, load_from_h5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_base_filename(spikedata: SpikeData, ) -> str:
    """
        Extract the base filename from a SpikeData object.

        Removes the `_preinfusion` and `_postinfusion` suffixes from the filename stem.

        Args:
            spikedata (SpikeData): SpikeData object containing the filename.

        Returns:
            str: Base filename without the infusion-related suffixes.
        """
    return spikedata.filename.stem.replace("_preinfusion", "").replace("_postinfusion", "")


def get_files(filepath: Path | str) -> tuple[SpikeData, ...]:
    """
        Retrieve SpikeData objects for all `.smr` files in the specified directory.

        Args:
            filepath (Path | str): Directory containing the `.smr` files.

        Returns:
            tuple[SpikeData, ...]: Tuple containing SpikeData objects for each `.smr` file.

        Raises:
            FileNotFoundError: If fewer than two `.smr` files are found in the directory.
    """
    file_list = list(Path(filepath).glob("*.smr"))
    if len(file_list) < 2:
        raise FileNotFoundError(
            f"Less than two files found in {filepath}"
        )
    return tuple(SpikeData(f) for f in file_list)


def concatenate_spike_data(spike_data_1: SpikeData, spike_data_2: SpikeData):
    """
        Concatenate data from two SpikeData objects.

        One SpikeData object should represent preinfusion data and the other postinfusion data.

        Args:
            spike_data_1 (SpikeData): First SpikeData object.
            spike_data_2 (SpikeData): Second SpikeData object.

        Returns:
            tuple: Four dictionaries containing concatenated LFP data, unit data,
                   LFP concatenation indices, and unit concatenation indices.

        Raises:
            ValueError: If both SpikeData objects are either preinfusion or postinfusion.
    """
    # Ensure one is preinfusion and the other is postinfusion
    if not ((spike_data_1.preinfusion and spike_data_2.postinfusion) or
            (spike_data_2.preinfusion and spike_data_1.postinfusion)):
        raise ValueError("One SpikeData object should be preinfusion and the other postinfusion.")

    # Determine the order of concatenation based on infusion type
    if spike_data_1.preinfusion:
        pre_spike_data = spike_data_1
        post_spike_data = spike_data_2
    else:
        pre_spike_data = spike_data_2
        post_spike_data = spike_data_1

    base_filename = get_base_filename(spike_data_1)

    combined_lfp = {}
    combined_unit = {}
    index_track_lfp = {}
    index_track_unit = {}

    # Concatenate lfp waveforms from pre- / post-infusion
    for key in pre_spike_data.lfp:
        combined_lfp[key] = np.concatenate([pre_spike_data.lfp[key], post_spike_data.lfp[key]])
        index_track_lfp[key] = len(pre_spike_data.lfp[key])

    # Concatenate unit waveforms from pre- / post-infusion
    for key in pre_spike_data.unit:
        combined_unit[key] = np.concatenate([pre_spike_data.unit[key], post_spike_data.unit[key]])
        index_track_unit[key] = len(pre_spike_data.unit[key])

    # Save to h5
    path = Path().home() / "data" / "combined" / f"{base_filename}.h5"
    save_to_h5(str(path), combined_unit, index_track_unit)
    logger.debug(f"Saved {base_filename}.h5")
    return combined_lfp, combined_unit, index_track_lfp, index_track_unit


def save_to_h5(filename: str,
               unit_dict: dict,
               idx_unit: dict
               ):
    """
        Save the specified data dictionaries to an HDF5 file.

        Args:
            filename (str): Path to the output HDF5 file.
            unit_dict (dict): Dictionary containing unit data.
            idx_unit (dict): Dictionary containing unit concatenation indices.
    """
    with h5py.File(filename, 'w') as f:
        # Save combined unit waveform array
        unit_group = f.create_group('unit')
        for key, data in unit_dict.items():
            unit_group.create_dataset(key, data=data)

        # Save index where unit was concatenated
        index_unit_group = f.create_group('idx_unit')
        for key, index in idx_unit.items():
            index_unit_group.create_dataset(key, data=index)

def read_group(group: h5py.Group) -> dict:
    data = {}
    # Read attributes
    for attr_name, attr_value in group.attrs.items():
        data[f"{attr_name}"] = attr_value
    # Read datasets and subgroups
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            data[key] = read_group(item)
        elif isinstance(item, h5py.Dataset):
            # item[()] reads the entire dataset into memory, similar to slicing with [:]
            data[key] = item[()]
    return data

def read_single_h5(filename: str | Path) -> dict:
    """
    Read a single HDF5 file and return a dictionary containing the data.

    Args:
    ----
        filename (str | Path): Path to the HDF5 file.

    Returns:
    -------
        dict: Dictionary containing the data from the HDF5 file.
    """
    with h5py.File(filename, "r") as f:
        data = read_group(f)
    return data

if __name__ == "__main__":
    datapath = Path().home() / "data"
    prepath = Path().home() / "data" / "smr"
    postpath = Path().home() / "data" / "combined"
    postpath.mkdir(exist_ok=True, parents=True)
    filenames = [x for x in datapath.glob("*.h5")]
    single_data = read_single_h5(filenames[0])
    second_data = read_single_h5(filenames[1])


    x = 5
