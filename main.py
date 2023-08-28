import numpy as np
import matplotlib.pyplot as plt
from math import floor

import tkinter as tk
import tkinter.filedialog

from sonpy import lib as sp

# Get file path
root = tk.Tk()
root.withdraw()

FilePath = tk.filedialog.askopenfilename()
print(FilePath)

# Open file
MyFile = sp.SonFile(FilePath, True)

if MyFile.GetOpenError() != 0:
    print('Error opening file:', sp.GetErrorString(MyFile.GetOpenError()))
    quit()

num_channels = sum(1 for i in range(MyFile.MaxChannels()) if
                   MyFile.ChannelType(i) != sp.DataType.Off and MyFile.ChannelType(i) == sp.DataType.Adc)

data = {}
for i in range(MyFile.MaxChannels()):
    file_time_base = MyFile.GetTimeBase()
    if MyFile.ChannelType(i) == sp.DataType.Adc:
        chan_type = MyFile.ChannelType(i)
        chan_title = MyFile.GetChannelTitle(i)

        chan_max_time = MyFile.ChannelMaxTime(i)
        chan_divide = MyFile.ChannelDivide(i)

        num_seconds = chan_max_time * file_time_base
        dPeriod = chan_divide * file_time_base
        nPoints2 = floor(num_seconds / dPeriod)
        chan_units = MyFile.GetChannelUnits(i)
        nPoints = floor(200 / dPeriod)
        # Read data
        wavedata = MyFile.ReadFloats(i, nPoints2, 0)

        data[chan_title] = {'chan_type': chan_type,
                            'chan_units': chan_units,
                            'chan_divide': chan_divide,
                            'chan_max_time': chan_max_time,
                            'wavedata': wavedata,
                            'nPoints': nPoints,
                            'nPoints2': nPoints2,
                            'dPeriod': dPeriod,
                            'num_seconds': num_seconds,
                            }

x = 5
