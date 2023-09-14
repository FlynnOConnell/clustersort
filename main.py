from math import floor
from pathlib import Path
import numpy as np
import h5py

from sonpy import lib as sp

filepath = Path().home() / 'data' / 'smr'
files = list(filepath.glob('*.smr'))

if not files or len(files) < 2:
    raise FileNotFoundError(f"No files found in {filepath} or less than two files found.")

sonfiles = [sp.SonFile(str(filename), True) for filename in files]
filedata = []
savepath = Path().home() / 'data' / 'h5'

base_names = [str(f.stem) for f in files]
common_prefixes = [name.rsplit('_', 1)[0] for name in base_names]


if common_prefixes[0] == common_prefixes[1]:
    hdf5_filename = common_prefixes[0] + '_combined.hdf5'
else:
    raise ValueError("The filenames before '_preinfusion' or '_postinfusion' are not identical.")

savename = savepath / hdf5_filename
exclude = ['Respirat', 'RefBrain', 'Sniff', ]

# Create the necessary directories if they don't exist
savepath.mkdir(parents=True, exist_ok=True)
for spkfile in sonfiles:
    num_channels = sum(1 for i in range(spkfile.MaxChannels()) if
                       spkfile.ChannelType(i) != sp.DataType.Off and spkfile.ChannelType(i) == sp.DataType.Adc)

    data = {}
    file_time_base = spkfile.GetTimeBase()

    for i in range(spkfile.MaxChannels()):
        channel_title = spkfile.GetChannelTitle(i)

        if spkfile.ChannelType(i) == sp.DataType.Adc and channel_title not in exclude and 'LFP' not in channel_title:
            chan_type = spkfile.ChannelType(i)
            chan_max_time = spkfile.ChannelMaxTime(i)
            chan_divide = spkfile.ChannelDivide(i)

            num_seconds = chan_max_time * file_time_base
            dPeriod = chan_divide * file_time_base
            nPoints2 = floor(num_seconds / dPeriod)
            chan_units = spkfile.GetChannelUnits(i)

            # Read data
            wavedata = spkfile.ReadFloats(i, nPoints2, 0)
            time = np.arange(0, len(wavedata) * dPeriod, dPeriod)

            data[channel_title] = {
                'chan_units': chan_units,
                'wavedata': wavedata,
                'sampling_rate': 1 / dPeriod,
                'num_seconds': num_seconds,
                'time': time,
            }

    filedata.append(data)

# Assuming filedata has two elements, each being a dictionary representing pre- and post-infusion data
pre_infusion_data = filedata[0]
post_infusion_data = filedata[1]

combined_data = {}
for key in pre_infusion_data.keys():
    # Ensure the key exists in both dictionaries before combining
    if key in post_infusion_data:
        # Combine wavedata arrays
        combined_wavedata = np.concatenate([pre_infusion_data[key]['wavedata'], post_infusion_data[key]['wavedata']])

        # Combine time arrays
        # For the post-infusion time array, we need to add the last time value of pre-infusion data
        # to all the time values to maintain continuity in the time series
        last_pre_infusion_time = pre_infusion_data[key]['time'][-1]
        adjusted_post_infusion_time = post_infusion_data[key]['time'] + last_pre_infusion_time

        combined_time = np.concatenate([pre_infusion_data[key]['time'], adjusted_post_infusion_time])

        # Now save this data to the new dictionary
        combined_data[key] = {
            'chan_units': pre_infusion_data[key]['chan_units'],  # Assuming chan_units remain same in both files
            'wavedata': combined_wavedata,
            'sampling_rate': pre_infusion_data[key]['sampling_rate'],
            'num_seconds': pre_infusion_data[key]['num_seconds'] + post_infusion_data[key]['num_seconds'],
            'time': combined_time,
        }

# Create a new HDF5 file
with h5py.File(savename, 'w') as f:
    # Loop over each channel in combined_data
    for channel_name, channel_data in combined_data.items():
        # Create a group for this channel
        grp = f.create_group(channel_name)

        # Save each array in this channel's data to a separate dataset within this group
        for key, value in channel_data.items():
            if key in ('wavedata', 'time'):  # These are NumPy arrays, so we can save them as datasets
                grp.create_dataset(key, data=value, compression="gzip", compression_opts=9)
            else:  # These are scalar values or strings, so we'll save them as attributes
                grp.attrs[key] = value


x = 5
