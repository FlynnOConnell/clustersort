from pathlib import Path
import h5py


def save_h5(filename, data, overwrite=False):
    """
    Save data to an HDF5 file. If the file already exists, it will be overwritten unless overwrite=False.
    If the file already exists and overwrite=False, the file will be saved with a new name, e.g. if
    "data.h5" already exists, it will be saved as "data_1.h5", incremented by 1 until a filename is not found.

    :param filename: Name of the file to save to
    :param data: Dictionary of data to save
    :param overwrite: Whether to overwrite the file if it already exists
    """
    filepath = Path(filename)
    if not overwrite and filepath.exists():
        i = 1
        while Path(filepath.stem + f"_{i}" + filepath.suffix).exists():
            i += 1
        filename = filepath.stem + f"_{i}" + filepath.suffix

    with h5py.File(filename, 'w') as f:
        for channel_name, channel_data in data.items():
            grp = f.create_group(channel_name)

            # Save each array in this channel's data to a separate dataset within this group
            for key, value in channel_data.items():
                if key in ('wavedata', 'time'):  # These are NumPy arrays, so we save them as datasets
                    grp.create_dataset(key, data=value, compression="gzip", compression_opts=9)
                else:  # These are scalar values or strings, so we'll save them as attributes
                    grp.attrs[key] = value

    print(f"Data saved to {filename}")


def read_h5(filename):
    """
    Read data from an HDF5 file.

    :param filename: Name of the file to read from
    :return: Dictionary of data read from the file
    """
    with h5py.File(filename, 'r') as f:
        data = {}
        for channel_name, channel_data in f.items():
            data[channel_name] = {}
            for key, value in channel_data.items():
                if key in ('wavedata', 'time'):
                    data[channel_name][key] = value[:]
                else:
                    data[channel_name][key] = value
    return data
