============
Installation
============

This pipeline requires Python 3.9+, and numpy <= 1.3.5 to comply with numba restrictions.

* `spk2py on GitHub <https://github.com/FlynnOConnell/spk2py/>`_

It is recommended to install using `mambaforge <https://mamba.readthedocs.io/en/latest/mamba-installation.html#mamba-install>`_ this will drastically speed up environment creation:

Installing from source
======================

**Linux and MacOS:**

.. code-block:: bash

    git clone https://github.com/FlynnOConnell/spk2py.git
    cd path/to/spk2py
    # This is for MambaForge, but you can use conda if you want
    # Note if you use conda, but want to go the mamba route, you will really want to uninstall miniconda/anaconda first
    wget -O Mambaforge.sh  "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
    bash Mambaforge.sh -b -p "${HOME}/conda"
    # !!!! FOR BASH USERS !!!!
    # If you dont know what these are, then use this one
    # If you use zsh, just change this to ~/.zshrc
    echo ". ${HOME}/conda/etc/profile.d/conda.sh" >> ~/.bashrc
    source "${HOME}/conda/etc/profile.d/conda.sh"

.. code-block:: bash

    mamba env create -f environment.yml # this will take a while
    conda activate spk2py
    pip install -r requirements.txt
    pip install -e .

Additionally, though not recommended, ``spk2py`` can be installed directly from pip:

.. warning::
   pip installing has **not** been tested on systems other than linux.
   Using ``mamba`` has been tested on each platform.
   As has docker.

.. code-block:: bash

    pip install spk2py

Mamba Installation
==================

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
*************

In addition to the Mambaforge standalone distribution (see above), there are also the
`condaforge/mambaforge <https://hub.docker.com/r/condaforge/mambaforge>`_ docker
images:

.. code-block:: bash

  docker run -it --rm condaforge/mambaforge:latest mamba info
