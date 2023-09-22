from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from spk_io import save_to_h5, read_single_h5, read_single_h5
from spike_data import SpikeData, load_from_h5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_base_filename(name_data: SpikeData | dict, ) -> str:
    """
        Extract the base filename from a SpikeData object.

        Removes the `_preinfusion` and `_postinfusion` suffixes from the filename stem.

        Args:
            name_data (SpikeData | dict): SpikeData object or dict containing the filename metadata.

        Returns:
            str: Base filename without the infusion-related suffixes.
        """
    if isinstance(name_data, SpikeData):
        return name_data.filename.stem.replace("_preinfusion", "").replace("_postinfusion", "")
    elif isinstance(name_data, dict):
        return name_data['metadata']['filename'].replace("_preinfusion", "").replace("_postinfusion", "")


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


def concatenate_spike_data(file_1: str | Path, file_2: str | Path):
    file_1 = Path(file_1)
    file_2 = Path(file_2)
    # Read h5 files
    data1 = read_single_h5(file_1)
    data2 = read_single_h5(file_2)

    # Validate metadata and segment count
    if not validate_same_metadata(data1, data2):
        raise ValueError("Metadata mismatch.")
    if len(data1['unit']) != len(data2['unit']):
        raise ValueError("Segment count mismatch.")

    # Use metadata from the first file
    combined_metadata = data1['metadata']

    combined_spikes = {}
    # Concatenate segments
    for key in data1['unit']:
        combined_spikes[key] = np.concatenate([data1['unit'][key]["spikes"], data2['unit'][key]["spikes"]])
        combined_times = np.concatenate([data1['unit'][key]["times"], data2['unit'][key]["times"]])

    # Save combined data
    base_filename = get_base_filename(data1)
    combined_file = {'metadata': combined_metadata, 'data': combined_spikes}
    path = Path().home() / "data" / "combined" / f"{base_filename}.h5"
    save_to_h5(str(path), combined_file)
    logger.debug(f"Saved {base_filename}.h5")
    return combined_file


def validate_same_metadata(data1: dict, data2: dict):
    for key in data1['metadata']:
        val1 = data1['metadata'][key]
        val2 = data2['metadata'][key]
        # TODO: Isolate instead exactly which keys should be the same, and which can be distinct
        if key not in ['filename', 'infusion', 'max_time', 'recording_length']:
            if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
                if not np.array_equal(val1, val2):
                    logger.debug(f"Metadata mismatch: {key}")
                    return False
            else:
                if val1 != val2:
                    logger.debug(f"Metadata mismatch: {key}")
                    return False
    return True


if __name__ == "__main__":
    datapath = Path().home() / "data"
    prepath = Path().home() / "data" / "smr"
    postpath = Path().home() / "data" / "combined"
    postpath.mkdir(exist_ok=True, parents=True)
    fnames = [x for x in datapath.glob("*.h5")]
    combined_data = concatenate_spike_data(fnames[0], fnames[1])


    x = 5
