from __future__ import annotations

import h5py
from pathlib import Path
import numpy as np
import logging
from spike_data import SpikeData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_files(filepath: Path | str) -> tuple[SpikeData, ...]:
    file_list = list(Path(filepath).glob("*.smr"))
    if len(file_list) < 2:
        raise FileNotFoundError(
            f"Less than two files found in {filepath}"
        )
    return tuple(SpikeData(f) for f in file_list)


def concatenate_spike_data(spike_data_1: SpikeData, spike_data_2: SpikeData):
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

    combined_lfp = {}
    combined_unit = {}
    index_track_lfp = {}
    index_track_unit = {}

    # Concatenate lfp
    for key in pre_spike_data.lfp:
        combined_lfp[key] = np.concatenate([pre_spike_data.lfp[key], post_spike_data.lfp[key]])
        index_track_lfp[key] = len(pre_spike_data.lfp[key])

    # Concatenate unit
    for key in pre_spike_data.unit:
        combined_unit[key] = np.concatenate([pre_spike_data.unit[key], post_spike_data.unit[key]])
        index_track_unit[key] = len(pre_spike_data.unit[key])

    return combined_lfp, combined_unit, index_track_lfp, index_track_unit


def save_to_h5(filename: str,
               lfp_dict: dict,
               unit_dict: dict,
               idx_lfp: dict,
               idx_unit: dict
               ):
    with h5py.File(filename, 'w') as f:
        # Save combined lfp waveform array
        lfp_group = f.create_group('lfp')
        for key, data in lfp_dict.items():
            lfp_group.create_dataset(key, data=data)

        # Save combined unit waveform array
        unit_group = f.create_group('unit')
        for key, data in unit_dict.items():
            unit_group.create_dataset(key, data=data)

        # Save index where lfp was concatenated
        index_lfp_group = f.create_group('idx_lfp')
        for key, index in idx_lfp.items():
            index_lfp_group.create_dataset(key, data=index)

        # Save index where unit was concatenated
        index_unit_group = f.create_group('idx_unit')
        for key, index in idx_unit.items():
            index_unit_group.create_dataset(key, data=index)

if __name__ == "__main__":
    prepath = Path().home() / "data" / "smr"
    files = get_files(prepath)
    lfp, unit, lfp_idx, unit_idx = concatenate_spike_data(files[0], files[1])
    x = 5
