import os
import sys
import importlib
from pathlib import Path

source_path = os.path.abspath('.')
docs_path = os.path.abspath('../')
package_path = os.path.abspath('../../')
spk2py_package_path = os.path.abspath('../../spk2py')

sys.path.insert(0, source_path)
sys.path.insert(0, docs_path)
sys.path.insert(0, package_path)
sys.path.insert(0, spk2py_package_path)
os.environ['PYTHONPATH'] = package_path + ':' + os.environ.get('PYTHONPATH', '')
os.environ['PYTHONPATH'] = spk2py_package_path + ':' + os.environ.get('PYTHONPATH', '')


# Minimum version, enforced by sphinx
needs_sphinx = '4.3'

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------

sys.path.insert(0, os.path.abspath('../sphinxext'))
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_design',
    'sphinx.ext.githubpages'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# List of directories, relative to source directories, that shouldn't be searched
# for source files.
exclude_dirs = []

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

numpydoc_show_class_members = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_favicon = '_static/favicon.ico'
html_title = "spk2py Docs"
html_static_path = ['_static']
html_last_updated_fmt = '%b %d, %Y'
html_css_files = ["spk2py.css"]
html_context = {"default_mode": "dark"}
html_file_suffix = '.html'

# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------
intersphinx_mapping = {
    'neps': ('https://numpy.org/neps', None),
    'python': ('https://docs.python.org/3', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'imageio': ('https://imageio.readthedocs.io/en/stable', None),
    'skimage': ('https://scikit-image.org/docs/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'scipy-lecture-notes': ('https://scipy-lectures.org', None),
    'pytest': ('https://docs.pytest.org/en/stable', None),
    'numpy-tutorials': ('https://numpy.org/numpy-tutorials', None),
    'numpydoc': ('https://numpydoc.readthedocs.io/en/latest', None),
    'dlpack': ('https://dmlc.github.io/dlpack/latest', None)
}

# -----------------------------------------------------------------------------
# spk2py extensions
# -----------------------------------------------------------------------------

# If we want to do a phantom import from an XML file for all autodocs
phantom_import_file = 'dump.xml'

# Make numpydoc to generate plots for example sections
numpydoc_use_plots = True

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True
