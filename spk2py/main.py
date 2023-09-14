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




x = 5
