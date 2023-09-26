===================
Configuration Guide
===================

----------------------------
Function: ``default_config``
----------------------------

The ``default_config`` function initializes a configuration file with default settings and creates necessary directories for the run.

**Parameters:**

- ``path: Path``: The path where the configuration file will be saved.
- ``config_ver: int``: The version of the configuration to be used. Default is 5.

-------------------------------------
Configuration Sections and Parameters
-------------------------------------

run-settings
============

Parameters defining the software and hardware configurations.

- ``resort-limit``: Default is '3'.
- ``cores-used``: Default is '8'.
- ``weekday-run``: Default is '2'.
- ``weekend-run``: Default is '8'.
- ``run-type``: Default is 'Auto'.
- ``manual-run``: Default is '2'.

paths
=====

Defines various paths necessary for the script.

- ``run-path``: Path to directory for files to be processed.
- ``results-path``: Path to directory for results.
- ``completed-path``: Path where completed files will be moved.

clustering
==========

Parameters defining the clustering process.

- ``max-clusters``: Default is '7'.
- ``max-iterations``: Default is '1000'.
- ``convergence-criterion``: Default is '.0001'.
- ``random-restarts``: Default is '10'.
- ``l-ratio-cutoff``: Default is '.1'.

signal
======

Parameters for signal preprocessing and spike detection.

- ``disconnect-voltage``: Default is '1500'.
- ``max-breach-rate``: Default is '.2'.
- ``max-breach-count``: Default is '10'.
- ``max-breach-avg``: Default is '20'.
- ``intra-hpc_cluster-cutoff``: Default is '3'.

filtering
=========

Filtering parameters to isolate the frequency range of interest.

- ``low-cutoff``: Default is '600'.
- ``high-cutoff``: Default is '3000'.

spike
=====

Spike detection and extraction parameters.

- ``pre-time``: Default is '.2'.
- ``post-time``: Default is '.6'.
- ``sampling-rate``: Default is '20000'.

std-dev
=======

Standard deviation parameters for spike detection and artifact removal.

- ``spike-detection``: Default is '2.0'.
- ``artifact-removal``: Default is '10.0'.

pca
===

Parameters for principal component analysis (PCA).

- ``variance-explained``: Default is '.95'.
- ``use-percent-variance``: Default is '1'.
- ``principal-component-n``: Default is '5'.

post-process
============

Post-processing parameters.

- ``reanalyze``: Default is '0'.
- ``simple-gmm``: Default is '1'.
- ``image-size``: Default is '70'.
- ``temporary-dir``: Default is user's home directory followed by '/tmp_python'.

version
=======

Version control parameters.

- ``config-version``: Default is determined by the ``config_ver`` parameter passed to the ``default_config`` function.