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


def merge_files(filepath: Path | str, savepath: Path | str = None) -> None:
    """
    Merge two .smr files into one .hdf5 file.

    :param filepath: Path to the directory containing the .smr files
    :param savepath: Path to the directory where the .hdf5 file will be saved
    :return: None
    """
    start_time = time.time()
    filepath = Path(filepath)
    files = list(filepath.glob("*.smr"))

    if not files or len(files) < 2:
        raise FileNotFoundError(
            f"No files found in {filepath} or less than two files found."
        )

    sonfiles = [sp.SonFile(str(filename), True) for filename in files]
    files = [Path(f) for f in files]
    logger.info(f"Files to be merged: {files}")
    if not savepath:
        savepath = Path().home() / "autosort" / "h5"
        savepath.mkdir(parents=True, exist_ok=True)
    else:
        savepath = Path(savepath)
        savepath.mkdir(parents=True, exist_ok=True)

    base_names = [str(f.stem) for f in files]
    common_prefixes = [name.rsplit("_", 1)[0] for name in base_names]
    if common_prefixes[0] == common_prefixes[1]:
        hdf5_filename = common_prefixes[0] + "_combined.hdf5"
    else:
        raise ValueError(
            "The filenames before '_preinfusion' or '_postinfusion' are not identical."
        )

    savename = savepath / hdf5_filename

    # These are unused in the spike-sorting pipeline, so we'll exclude them
    exclude = [
        "Respirat",
        "RefBrain",
        "Sniff",
    ]

    filedata = []
    for spkfile in sonfiles:
        data = {}
        file_time_base = spkfile.GetTimeBase()
        for i in range(spkfile.MaxChannels()):
            channel_title = spkfile.GetChannelTitle(i)

            if (
                spkfile.ChannelType(i) == sp.DataType.Adc
                and channel_title not in exclude
                and "LFP" not in channel_title
            ):
                chan_max_time = spkfile.ChannelMaxTime(i)
                chan_divide = spkfile.ChannelDivide(i)

                recording_length = chan_max_time * file_time_base
                dPeriod = chan_divide * file_time_base
                num_ticks = floor(recording_length / dPeriod)
                chan_units = spkfile.GetChannelUnits(i)

                # Read data
                wavedata = spkfile.ReadFloats(i, num_ticks, 0)
                # time = np.arange(0, len(wavedata) * dPeriod, dPeriod) #  Need this later

                data[channel_title] = {
                    "chan_units": chan_units,
                    "wavedata": wavedata,
                    "sampling_rate": 1 / dPeriod,
                    "recording_length": recording_length,
                }
        filedata.append(data)

    pre_infusion_data = filedata[0]
    post_infusion_data = filedata[1]
    combined_data = {}
    for key in pre_infusion_data.keys():
        # Ensure the key exists in both dictionaries before combining
        if key in post_infusion_data:
            # Combine wavedata arrays
            combined_wavedata = np.concatenate(
                [
                    pre_infusion_data[key]["wavedata"],
                    post_infusion_data[key]["wavedata"],
                ]
            )

            # Combine time arrays
            # For the post-infusion time array, we need to add the last time value of pre-infusion data
            # to all the time values to maintain continuity in the time series
            # last_pre_infusion_time = pre_infusion_data[key]['time'][-1]
            # adjusted_post_infusion_time = post_infusion_data[key]['time'] + last_pre_infusion_time
            # combined_time = np.concatenate([pre_infusion_data[key]['time'], adjusted_post_infusion_time])

            # Now save this data to the new dictionary
            combined_data[key] = {
                "chan_units": pre_infusion_data[key][
                    "chan_units"
                ],
                "wavedata": combined_wavedata,
                "sampling_rate": pre_infusion_data[key]["sampling_rate"],
                "recording_length": pre_infusion_data[key]["recording_length"]
                + post_infusion_data[key]["recording_length"],
                # 'time': combined_time,
            }
    h5.save_h5(savename, combined_data, overwrite=True)
    logger.info(f"Files merged in {time.time() - start_time} seconds.")
    logger.info(f"{len(sonfiles)} files saved to {savename}")


if __name__ == "__main__":
    mergepath = Path().home() / "data"
    merge_files(mergepath)
