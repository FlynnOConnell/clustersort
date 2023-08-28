from neo.io.spike2io import Spike2IO
import numpy as np
import matplotlib.pyplot as plt

# Load Spike2 file
reader = Spike2IO(filename=r'C://Users//Flynn//Dropbox//Cornell//5HTephys_R11_cinanserin_120321_preinfusion.smr')
bl = reader.read(lazy=False)[0]
print(bl)
# access to segments
for seg in bl.segments:
    for asig in seg.analogsignals:
        print(f"analog: {asig}")
    for st in seg.spiketrains:
        print(f"st: {st}")

x = 5