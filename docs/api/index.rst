Usage
=====

The first step is to create an instance of the :class:`~autosort.directory_manager.DirectoryManager` class and the :class:`~autosort.spk_config.SpkConfig` class:

.. autosummary::
    autosort.DirectoryManager
    autosort.SpkConfig

.. note::

    The ``autosort`` package is designed to be used with the ``autosort`` command line tool. See
    :ref:`dir_guide` for more information.

Directory Management
--------------------

.. currentmodule:: autosort.directory_manager

.. automodule:: directory_manager
    :members:
    :undoc-members:
    :show-inheritance:


Configuration
-------------

.. currentmodule:: autosort.spk_config

.. autoclass:: SpkConfig

The table below indicates which ``Process`` functions are supported on the CPU (e.g. using
``pandas``), on CPU with Dask (e.g. using ``dask.dataframe``), on the GPU (e.g. using ``cudf``),
and on the GPU with Dask (e.g. using ``dask-cudf``). The final two columns indicate which reductions
support antialiased lines and which can be used as the ``selector`` in a
:class:`~autosort.autosort.Process` task.

.. csv-table::
   :file: reduction.csv
   :header-rows: 1
