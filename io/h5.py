import os
import numpy as np
import h5py


# Create EArrays in hdf5 file
def create_hdf_arrays(file_name, units=None, lfp=None):
    hf5 = h5py.open_file(file_name, 'r+')
    atom = tables.IntAtom()

    # Create arrays unit and lfp data
    for i in units:
        ea_units = hf5.create_earray('/units', 'units_%i' % i, atom, (0,))

    for i in lfp:
        ea_lfp = hf5.create_earray('/lfp', 'lfp_%i' % i, atom, (0,))

    # Close the hdf5 file
    hf5.close()


# Read files into hdf5 arrays - the format should be 'one file per channel'
def read_files(hdf5_name, ports, dig_in, emg_port, emg_channels):
    hf5 = tables.open_file(hdf5_name, 'r+')

    # Read digital inputs, and append to the respective hdf5 arrays
    for i in dig_in:
        inputs = np.fromfile('board-DIN-%02d' % i + '.dat', dtype=np.dtype('uint16'))
        exec("hf5.root.digital_in.dig_in_" + str(i) + ".append(inputs[:])")

    # Read data from amplifier channels
    emg_counter = 0
    el_counter = 0
    for port in ports:
        for channel in range(32):
            data = np.fromfile('amp-' + port + '-%03d' % channel + '.dat', dtype=np.dtype('int16'))
            if port == emg_port[0] and channel in emg_channels:
                exec("hf5.root.raw_emg.emg%i.append(data[:])" % emg_counter)
                emg_counter += 1
            else:
                exec("hf5.root.raw.electrode%i.append(data[:])" % el_counter)
                el_counter += 1
        hf5.flush()

    hf5.close()
