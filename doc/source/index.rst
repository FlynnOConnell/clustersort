===========================
Welcome to spk2py's Documentation
===========================

.. note::
   This project, spk2py, aims to provide a Python-based framework for processing and analyzing electrophysiological spike data. It includes capabilities for reading HDF5 files, parallel processing, and configurable runs through an external configuration setup.

Overview
========

* Introduction
* Features
* Installation

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :name: gettingstarted

   install


Core Modules
============

Auto Sort
---------
.. automodule:: spk2py.autosort.autosort
   :members:

.. automodule:: spk2py.autosort.autosort_post
   :members:

.. automodule:: spk2py.autosort.directory_manager
   :members:

.. automodule:: spk2py.autosort.main
   :members:

.. automodule:: spk2py.autosort.spk_config
   :members:

Cluster
-------
.. automodule:: spk2py.cluster
   :members:

Merge
-----
.. automodule:: merge
   :members:

Spike Data
----------
.. automodule:: spike_data
   :members:

I/O Operations
--------------
.. automodule:: spk_io
   :members:

Logging
-------
.. automodule:: spk_logging.logger_config
   :members:

Tests
-----
.. automodule:: tests.merge
   :members:

.. automodule:: tests.signal
   :members:

.. automodule:: tests.sort
   :members:


Technical Explanation
=====================

* Spike Waveform Samples

.. toctree::
   :maxdepth: 2
   :caption: Technical Explanation
   :name: technical

   spike_waveform_samples_explanation

Indices and Tables
==================

.. toctree::
   :maxdepth: 2
   :caption: Additional Resources

