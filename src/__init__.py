"""
==========
``clustersort``
==========

*spk2py. A full analysis pipeline for spike preprocessing, sorting and post-processing utilities.*

spk2py is a Python-based framework for processing and analyzing electrophysiological spike data.
It extracts spike waveforms from raw Spike2 files using the C++ wrapper SonPy library, and then
performs waveform extraction and spike sorting. It also provides a number of tools for analyzing
the resulting spike data. Once spikes are sorted, users are able to post-process the spikes using
plots for mahalanobis distance, ISI, and autocorrelograms. The spike data can also be exported to
a variety of formats for further analysis in other programs.

Documentation Guide
-------------------

I recommend exploring the docstrings using
`IPython <https://ipython.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.

"""
from spk2py import util
from spk2py.logger import logger_config

__all__ = ["util", "logger_config"]
