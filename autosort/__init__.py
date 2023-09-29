"""
===============
``clustersort``
===============

*clustersort. A fully automated and configurable spike sorting pipeline for extracellular recordings.*

A utility for cluster-analysis spike sorting with electrophysiological data.
Once spikes are sorted, users are able to post-process the spikes using
plots for mahalanobis distance, ISI, and autocorrelograms. The spike data can also be exported to
a variety of formats for further analysis in other programs.

The pipeline is as follows:

1. Read in the data from the pl2 file.
2. Filter the data.
3. Extract spikes from the filtered data.
4. Cluster the spikes.
5. Perform breach analysis on the clusters.
6. Resort the clusters based on the breach analysis.
7. Save the data to an HDF5 file, and graphs to given plotting folders.

Documentation Guide
-------------------

I recommend exploring the docstrings using
`IPython <https://ipython.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.

"""
from packaging.version import Version

from .directory_manager import DirectoryManager  # noqa: (API import)
from .spk_config import SpkConfig  # noqa: (API import)
from .utils import *  # noqa: (API import)
from .logger import *  # noqa: (API import)
from .main import *  # noqa: (API import)
from .sort import *  # noqa: (API import)

from numpy import __version__ as numpyversion

if Version(numpyversion) >= Version("1.24.0"):
    raise ImportError(
        "numpy version 1.24.0 or greater is not supported due to numba incompatibility, "
        "please downgrade to 1.23.5 or lower"
    )
