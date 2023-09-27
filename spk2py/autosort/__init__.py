"""
========
Autosort
========
.. currentmodule:: spk2py.autosort

.. autosummary::
    :toctree: generated/

Autosort: A Python package for spike sorting.

This package is designed to automate the spike sorting process for extracellular recordings.

It is designed to be used with the Spike2 software package from CED, with pl2 files from Plexon
in development.

The package is designed to be used with the AutoSort pipeline, which is a series of steps that
are performed on the data to extract spikes and cluster them. The pipeline is as follows:

1. Read in the data from the pl2 file.
2. Filter the data.
3. Extract spikes from the filtered data.
4. Cluster the spikes.
5. Perform breach analysis on the clusters.
6. Resort the clusters based on the breach analysis.
7. Save the data to an HDF5 file, and graphs to given plotting folders.

"""
from .autosort import *
from .directory_manager import *
from .spk_config import *
from .wf_shader import *

__all__ = [
    "run_spk_process",
    "ProcessChannel",
    "DirectoryManager",
    "SpkConfig",
    "waveforms_datashader",
]
