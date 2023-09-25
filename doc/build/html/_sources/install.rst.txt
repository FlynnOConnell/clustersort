============
Installation
============

This pipeline requires Python 3.9+, and numpy <= 1.3.5 to comply with numba restrictions.

* `spk2py on GitHub <https://github.com/FlynnOConnell/spk2py/>`_

It is recommended to install using `mambaforge <https://mamba.readthedocs.io/en/latest/mamba-installation.html#mamba-install>`_ this will drastically speed up environment creation:

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
