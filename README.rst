Clustersort - Semi-Supervised Spike Sorting
===========================================

This Python repository is adapted from the methods and code described in Mukherjee, Wachutka, & Katz (2017) [1]_.
A large percentage of the clustering parameters were made in reference to Reddish (2005) [2]_.

Heavily inspired by `blechpy <https://github.com/nubs01/blechpy>`_.

.. note::

   This program is designed for sorting spikes from electrophysiological recordings into single, isolated units. The primary input is a .h5 file containing the continuous signal or thresholded waveforms.
   Alternatively, numpy arrays and pandas dataframes can be used with the help of data-loaders.

**Compatibility and Licensing**
This software is compatible with Windows, macOS, and Linux, and is well-suited for containerization and high-performance computing clusters. It is distributed under the GNU General Public License v3.0 (GPLv3). For more information, consult the LICENSE file in this repository.


Sort Criteria
-------------
The primary criteria for considering a unit isolated are:

#. 1 ms ISIs must be <= 0.5%
#. The waveform must be cellular
#. The unit must be sufficiently separated based on Mahalanobis distribution
#. L-Ratio must be <= 0.1, as described in Schmitzer-Torbert et al. (2005) [2]_.


.. image:: https://img.shields.io/badge/view-Documentation-blue?style=
   :alt: Go to project documentation
   :target: https://flynnoconnell.github.io/clustersort/index.html#
.. image:: https://readthedocs.org/projects/clustersort/badge/?version=latest
   :alt: Documentation Status
   :target: https://clustersort.readthedocs.io/en/latest/?badge=latest
.. image:: https://dl.circleci.com/status-badge/img/gh/FlynnOConnell/clustersort/tree/master.svg?style=shield
   :alt: CircleCI

Default File structure
-----------------------

::

    ~/
    ├── autosort
    │   ├── h5
    │   ├── completed
    │   ├── results
    │   ├── to_run
    │   └── autosort_config.ini


To initialize a configuration file with default settings, use the `default_config` function as follows:

.. code-block:: python
    from pathlib import Path
    from clustersort.spk_config import default_config

    default_config(path=Path('/path/to/your/config.ini'))

.. _config-module:

Installation
============

This pipeline requires Python 3.9+, and numpy <= 1.3.5 to comply with numba restrictions.

It is *highly-recommended* to install using `mambaforge <https://mamba.readthedocs.io/en/latest/mamba-installation.html#mamba-install>`_ this will drastically speed up environment creation:

Installing from source
----------------------

**Linux and MacOS:**

.. code-block:: bash

    git clone https://github.com/FlynnOConnell/clustersort.git
    cd path/to/clustersort
    # This is for MambaForge, but you can use conda if you want
    # Note if you use conda, but want to go the mamba route, you will really want to uninstall miniconda/anaconda first
    wget -O Mambaforge.sh  "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
    bash Mambaforge.sh -b -p "${HOME}/conda"
    # !!!! FOR BASH USERS !!!!
    # If you dont know what these are, then use this one
    # If you use zsh, just change this to ~/.zshrc
    echo ". ${HOME}/conda/etc/profile.d/conda.sh" >> ~/.bashrc
    source "${HOME}/conda/etc/profile.d/conda.sh"


If you're getting ``conda: command not found``, you need to add ``conda`` to your path.
Look in your home directory, you should have a mambaforge or miniforge3 folder, depending on
your method of installation. Add that folder/bin to your path:
`export PATH="/home/username/mambaforge/bin:$PATH"`

.. code-block:: bash

    mamba env create -f environment.yml # this will take a while
    conda activate clustersort
    pip install -r requirements.txt
    pip install -e .

Additionally, though not recommended, ``clustersort`` can be installed directly from pip:

.. warning::
   pip installing has **not** been tested on systems other than linux.
   Using ``mamba`` has been tested on each platform.
   As has docker.

.. code-block:: bash

    pip install clustersort


Mamba Installation
------------------

We recommend that you start with the `Mambaforge distribution <https://github.com/conda-forge/miniforge#mambaforge>`_.
Mambaforge comes with the popular ``conda-forge`` channel preconfigured, but you can modify the configuration to use any channel you like.
Note that Anaconda channels are generally incompatible with conda-forge, so you should not mix them.

.. note::
   For both ``mamba`` and ``conda``, the ``base`` environment is meant to hold their dependencies.
   It is strongly discouraged to install anything else in the base envionment.
   Doing so may break ``mamba`` and ``conda`` installation.


Existing ``conda`` install (not recommended)
********************************************

.. warning::
   This way of installing Mamba is **not recommended**.
   We strongly recommend to use the Mambaforge method (see above).

To get ``mamba``, just install it *into the base environment* from the ``conda-forge`` channel:

.. code:: bash

   # NOT RECOMMENDED: This method of installation is not recommended, prefer Mambaforge instead (see above)
   # conda install -n base --override-channels -c conda-forge mamba 'python_abi=*=*cp*'


.. warning::
   Installing mamba into any other environment than ``base`` is not supported.


Docker images
-------------

In addition to the Mambaforge standalone distribution (see above), there are also the
`condaforge/mambaforge <https://hub.docker.com/r/condaforge/mambaforge>`_ docker
images:

.. code-block:: bash

  docker run -it --rm condaforge/mambaforge:latest mamba info

.. _wf_sample:


Configuration Guide
=========================

Sections
--------

.. _run-section:

run
~~~
Configuration parameters for the runtime of the pipeline.

    .. _run-resort-limit-key:

    - resort-limit
        - Description: The maximum number of times the sorting process can be rerun.
        - Default: 3

    .. _run-cores-used-key:

    - cores-used
        - Description: The number of cores to be used during the run.
        - Default: 8

    .. _run-weekday-run-key:

    - weekday-run
        - Description: The number of runs allowed on a weekday.
        - Default: 2

    .. _run-weekend-run-key:

    - weekend-run
        - Description: The number of runs allowed on a weekend.
        - Default: 8

    .. _run-run-type-key:

    - run-type
        - Description: Defines the type of run (Auto/Manual).
        - Default: Auto

    .. _run-manual-run-key:

    - manual-run
        - Description: The number of manual runs allowed.
        - Default: 2

.. _path-section:

path
~~~~
Here we define various paths necessary for the script, set by default to subdirectories in the parent directory of the specified path.

    .. _path-run-path-key:

    - run-path
        - Description: Path to the directory where files to be processed are stored.
        - Default: None specified

    .. _path-results-path-key:

    - results-path
        - Description: Path to the directory where results will be stored.
        - Default: None specified

    .. _path-completed-path-key:

    - completed-path
        - Description: Path where completed files will be moved.
        - Default: None specified

.. _cluster-section:

cluster
~~~~~~~
Parameters defining the clustering process:

    .. _cluster-max-clusters-key:

    - max-clusters
        - Description: Maximum number of clusters to use in the clustering algorithm.
        - Default: 7

    .. _cluster-max-iterations-key:

    - max-iterations
        - Description: Maximum number of iterations for the clustering algorithm.
        - Default: 1000

    .. _cluster-convergence-criterion-key:

    - convergence-criterion
        - Description: The criterion for convergence in the clustering algorithm.
        - Default: .0001

    .. _cluster-random-restarts-key:

    - random-restarts
        - Description: Number of random restarts in the clustering process to avoid local minima.
        - Default: 10

    .. _cluster-l-ratio-cutoff-key:

    - l-ratio-cutoff
        - Description: The cutoff value for the L-Ratio metric, used to assess cluster quality.
        - Default: .1

.. _breach-section:

breach
~~~~~~
Parameters involved in signal preprocessing and spike detection:

    .. _breach-disconnect-voltage-key:

    - disconnect-voltage
        - Description: Voltage level that indicates a disconnection in the signal, to detect noise or artifacts.
        - Default: 1500

    .. _breach-max-breach-rate-key:

    - max-breach-rate
        - Description: The maximum rate at which breaches (potentially signal artifacts or spikes) can occur before it is considered noise.
        - Default: .2

    .. _breach-max-breach-count-key:

    - max-breach-count
        - Description: The maximum count of breaches allowed in a given window of time.
        - Default: 10

    .. _breach-max-breach-avg-key:

    - max-breach-avg
        - Description: Perhaps the average breach level over a defined window.
        - Default: 20

    .. _breach-intra-hpc_cluster-cutoff-key:

    - intra-hpc_cluster-cutoff
        - Description: A cutoff value for considering a signal as noise based on some intra-cluster metric.
        - Default: 3

.. _filter-section:

filter
~~~~~~
Filtering parameters to isolate the frequency range of interest:

    .. _filter-low-cutoff-key:

    - low-cutoff
        - Description: The low cutoff frequency for a band-pass filter.
        - Default: 600

    .. _filter-high-cutoff-key:

    - high-cutoff
        - Description: The high cutoff frequency for the band-pass filter.
        - Default: 3000

.. _spike-section:

spike
~~~~~
Spike detection and extraction parameters:

    .. _spike-pre-time-key:

    - pre-time
        - Description: Time before a spike event to include in each spike waveform, in seconds.
        - Default: .2

    .. _spike-post-time-key:

    - post-time
        - Description: Time after a spike event to include in each spike waveform, in seconds.
        - Default: .6

    .. _spike-sampling-rate-key:

    - sampling-rate
        - Description: The sampling rate of the recording, in Hz.
        - Default: 20000

.. _detection-section:

detection
~~~~~~~~~
Standard deviation parameters for spike detection and artifact removal:

    .. _detection-spike-detection-key:

    - spike-detection
        - Description: A multiplier for the standard deviation of the noise to set a threshold for spike detection.
        - Default: 2.0

    .. _detection-artifact-removal-key:

    - artifact-removal
        - Description: A threshold for artifact removal, based on a multiple of the standard deviation.
        - Default: 10.0

.. _pca-section:

pca
~~~
Parameters defining how principal component analysis (PCA) is conducted on the spike waveforms:

    .. _pca-variance-explained-key:

    - variance-explained
        - Description: The proportion of variance explained to determine the number of principal components to retain.
        - Default: .95

    .. _pca-use-percent-variance-key:

    - use-percent-variance
        - Description: Whether to use percent variance to determine the number of components to retain.
        - Default: 1

    .. _pca-principal-component-n-key:

    - principal-component-n
        - Description: An alternative to variance-explained, specifying the number of principal components to retain directly.
        - Default: 5

.. _postprocess-section:

postprocess
~~~~~~~~~~~
Post-processing parameters:

    .. _postprocess-reanalyze-key:

    - reanalyze
        - Description: Whether to reanalyze the data.
        - Default: 0

    .. _postprocess-simple-gmm-key:

    - simple-gmm
        - Description: Whether to use a simple Gaussian Mixture Model in the post-processing.
        - Default: 1

    .. _postprocess-image-size-key:

    - image-size
        - Description: The size of images generated during post-processing.
        - Default: 70

    .. _postprocess-temporary-dir-key:

    - temporary-dir
        - Description: The directory to store temporary files during processing.
        - Default: user's home directory followed by '/tmp_python'

