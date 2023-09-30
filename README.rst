===========================================
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

```python
from pathlib import Path
from clustersort.spk_config import default_config

default_config(path=Path('/path/to/your/config.ini'))
```

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

Spike Waveform Sampling
=======================

When processing electrophysiological data to extract spike waveforms, it's crucial to understand how many samples each extracted spike waveform will contain.

Setting Up Snapshot Duration
----------------------------

In the extract_waveforms() code, a "snapshot" around each detected spike is defined using the parameter `spike_snapshot`, set to `(0.2, 0.6)` milliseconds by default. This means that each snapshot will contain:

- 0.2 milliseconds of data before the spike
- 0.6 milliseconds of data after the spike
- The spike itself

Calculating Number of Samples
-----------------------------

Given a sampling rate of 18,518 Hz, the duration of each sample would be :math:`\frac{1}{18518}` seconds or approximately 0.054 milliseconds.

1. Pre-spike samples: :math:`\frac{0.2}{0.054} \approx 3.7`, rounded to 4 samples
2. Post-spike samples: :math:`\frac{0.6}{0.054} \approx 11.1`, rounded to 11 samples
3. Adding 1 for the spike itself gives you :math:`4 + 11 + 1 = 16` samples

Configuration Guide
===================

Overview
--------

This class manages configurations for the clustersort pipeline. It reads from an INI-style configuration file and provides methods to access configurations for various sections.

.. moduleauthor:: Flynn O'Connell


Initialize a new ``SpkConfig`` object by specifying the ``cfg_path`` parameter. If no path is provided, it defaults to a predefined location.

Attributes
----------

- ``cfg_path``: Path to the configuration file, either provided by the user or a default path.
- ``config``: A ``ConfigParser`` object containing the loaded configurations.
- ``params``: A dictionary containing all the configuration parameters.

Methods
-------

get_section(section: str)
    Returns a dictionary containing key-value pairs for the given section.

set(section: str, key: str, value: Any)
    Sets a value for a configuration parameter within a specified section.

get_all()
    Returns a dictionary containing all key-value pairs from all sections.

Property Methods
----------------

run
    Returns a dictionary containing key-value pairs for the 'run' section.

path
    Returns a dictionary containing key-value pairs for the 'path' section.

cluster
    Returns a dictionary containing key-value pairs for the 'cluster' section.

breach
    Returns a dictionary containing key-value pairs for the 'breach' section.

filter
    Returns a dictionary containing key-value pairs for the 'filter' section.

spike
    Returns a dictionary containing key-value pairs for the 'spike' section.

detection
    Returns a dictionary containing key-value pairs for the 'detection' section.

pca
    Returns a dictionary containing key-value pairs for the 'pca' section.

postprocess
    Returns a dictionary containing key-value pairs for the 'postprocess' section.

INI Configuration File
----------------------

This file is the easiest entrypoint to change parameters. You can specify where this file
is created with the ``cfg_path`` attribute.

- ``run``: Contains runtime settings like ``resort-limit``, ``cores-used``.
- ``path``: Contains path settings like directories for ``run``, ``results``.
- ``cluster``: Contains clustering parameters like ``max-clusters``, ``max-iterations``.
- ``breach``: Contains breach analysis parameters like ``disconnect-voltage``, ``max-breach-rate``.
- ``filter``: Contains filter parameters like ``low-cutoff``, ``high-cutoff``.
- ``spike``: Contains spike-extraction settings like ``pre-time``, ``post-time``.

Note: All values are stored as strings due to the nature of INI files. It's up to the user to convert these to appropriate types.

Example
-------

.. code-block:: python

    cfg = SpkConfig()
    run = cfg.run
    print(type(run), run)

    cfg.set('run', 'resort-limit', 5)
    print(cfg.run['resort-limit'])

See Also
--------

- `configparser from python std library <https://docs.python.org/3/library/configparser.html>`_


See Also
========

- `configparser from python std library <https://docs.python.org/3/library/configparser.html>`_

References
==========

.. [1] Mukherjee, Narendra & Wachutka, Joseph & Katz, Donald. (2017). Python meets systems neuroscience: affordable, scalable and open-source electrophysiology in awake, behaving rodents. 98-105.

.. [2] Schmitzer-Torbert N, Jackson J, Henze D, Harris K, Redish AD. Quantitative measures of cluster quality for use in extracellular recordings. Neuroscience. 2005;131:1–11.

