from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import h5py

Segment = namedtuple("Segment", ["segment_number", "data"])
UnitData = namedtuple("UnitData", ["slices", "times"])

def save_to_h5(filename: str,
               unit_dict: dict,
               ):
    """
        Save the specified data dictionaries to an HDF5 file.

        Args:
            filename (str): Path to the output HDF5 file.
            unit_dict (dict): Dictionary containing unit data.
    """
    with h5py.File(filename, 'w') as f:
        # Save combined unit waveform array
        unit_group = f.create_group('unit')
        for key, data in unit_dict.items():
            unit_group.create_dataset(key, data=data)

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
