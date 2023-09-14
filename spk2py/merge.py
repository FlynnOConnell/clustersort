from __future__ import annotations

from math import floor
from pathlib import Path
import numpy as np
import h5py
from .io import h5

from sonpy import lib as sp


def merge_files(filepath: Path | str):
    """
    Merge two .smr files into one .hdf5 file.

    :param filepath: Path to the directory containing the .smr files
    :return: None
    """
    filepath = Path(filepath)
    files = list(filepath.glob('*.smr'))

    if not files or len(files) < 2:
        raise FileNotFoundError(
            f"No files found in {filepath} or less than two files found."
        )

    sonfiles = [sp.SonFile(str(filename), True) for filename in files]
    savepath = Path().home() / 'data' / 'h5'
    savepath.mkdir(parents=True, exist_ok=True)

    # Get the common prefix of the filenames for saving
    base_names = [str(f.stem) for f in files]
    common_prefixes = [name.rsplit('_', 1)[0] for name in base_names]
    if common_prefixes[0] == common_prefixes[1]:
        hdf5_filename = common_prefixes[0] + '_combined.hdf5'
    else:
        raise ValueError("The filenames before '_preinfusion' or '_postinfusion' are not identical.")

    savename = savepath / hdf5_filename

    # These are unused in the spike-sorting pipeline, so we'll exclude them
    exclude = ['Respirat', 'RefBrain', 'Sniff', ]

    filedata = []
    for spkfile in sonfiles:

        data = {}
        file_time_base = spkfile.GetTimeBase()
        for i in range(spkfile.MaxChannels()):
            channel_title = spkfile.GetChannelTitle(i)

            if (spkfile.ChannelType(i) == sp.DataType.Adc
                    and channel_title not in exclude
                    and 'LFP' not in channel_title):
                chan_max_time = spkfile.ChannelMaxTime(i)
                chan_divide = spkfile.ChannelDivide(i)

                num_seconds = chan_max_time * file_time_base
                dPeriod = chan_divide * file_time_base
                nPoints2 = floor(num_seconds / dPeriod)
                chan_units = spkfile.GetChannelUnits(i)

                # Read data
                wavedata = spkfile.ReadFloats(i, nPoints2, 0)
                # time = np.arange(0, len(wavedata) * dPeriod, dPeriod) #  Need this later

                data[channel_title] = {
                    'chan_units': chan_units,
                    'wavedata': wavedata,
                    'sampling_rate': 1 / dPeriod,
                    'num_seconds': num_seconds,
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
                [pre_infusion_data[key]['wavedata'], post_infusion_data[key]['wavedata']])

            # Combine time arrays
            # For the post-infusion time array, we need to add the last time value of pre-infusion data
            # to all the time values to maintain continuity in the time series
            # last_pre_infusion_time = pre_infusion_data[key]['time'][-1]
            # adjusted_post_infusion_time = post_infusion_data[key]['time'] + last_pre_infusion_time
            # combined_time = np.concatenate([pre_infusion_data[key]['time'], adjusted_post_infusion_time])

            # Now save this data to the new dictionary
            combined_data[key] = {
                'chan_units': pre_infusion_data[key]['chan_units'],  # Assuming chan_units remain same in both files
                'wavedata': combined_wavedata,
                'sampling_rate': pre_infusion_data[key]['sampling_rate'],
                'num_seconds': pre_infusion_data[key]['num_seconds'] + post_infusion_data[key]['num_seconds'],
                # 'time': combined_time,
            }
    h5.save_h5(savename, combined_data, overwrite=True)
    print(f'Files saved to: {savename}')
