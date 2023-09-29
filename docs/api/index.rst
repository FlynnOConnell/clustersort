.. currentmodule:: clustersort

Usage
=====

1. Configuration
****************

The clustersort pipeline manages directories for 2 primary purposes:

- an entrypoint for the data being sorted
- a place to store plots and temporary data files

Both of these use cases are handled via the :class:`~directory_manager.DirectoryManager`.

For the actual sorting, there are a variety of paramters that can be tuned based on the type
of data the user is working with. This is managed by :class:`~spk_config.SpkConfig`. These parameters are covered in detail in the clustersort
configuration guide. :ref:`config <config_guide>`

.. admonition:: Example

   A structured data type containing a 16-character string (in field 'name')
   and a sub-array of two 64-bit floating-point number (in field 'grades'):

   >>> dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
   >>> dt['name']
   dtype('<U16')
   >>> dt['grades']
   dtype(('<f8', (2,)))

   Items of an array of this data type are wrapped in an :ref:`array
   scalar <arrays.scalars>` type that also has two fields:

   >>> x = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dtype=dt)
   >>> x[1]
   ('John', [6., 7.])
   >>> x[1]['grades']
   array([6.,  7.])
   >>> type(x[1])
   <class 'numpy.void'>
   >>> type(x[1]['grades'])
   <class 'numpy.ndarray'>

.. autosummary::
   :toctree: generated/

   DirectoryManager
   SpkConfig
   ProcessChannel
   cluster_gmm
   run
   configure_logger

