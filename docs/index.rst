=================================
Welcome to spk2py's Documentation
=================================

.. note::
    spk2py is a Python-based framework for processing and analyzing electrophysiological spike data.
    It extracts spike waveforms from raw Spike2 files using the lower-level SonPy library, and then
    performs waveform extraction and spike sorting. It also provides a number of tools for analyzing
    the resulting spike data. Once spikes are sorted, users are able to post-process the spikes using
    plots for mahalanobis distance, ISI, and autocorrelograms. The spike data can also be exported to
    a variety of formats for further analysis in other programs.


Overview
========

.. toctree::
   :maxdepth: 4
   :caption: Getting Started
   :name: gettingstarted

   install


Configuration
=============

.. toctree::
   :maxdepth: 2
   :caption: Configuration Guide
   :name: configuration

   configuration_guide

Technical Explanation
=====================

.. toctree::
   :maxdepth: 2
   :caption: Technical Explanation
   :name: technical

   spike_waveform_samples_explanation

Users Guide
===========

.. toctree::
   :maxdepth: 2
   :caption: Users Guide
   :name: guide

   guide/autosort

API Documentation
=================

.. toctree::
   :maxdepth: 2
   :caption: API
   :name: api

   spike_data

