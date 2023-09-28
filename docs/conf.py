# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import importlib
from pathlib import Path

source_path = os.path.abspath('.')
docs_path = os.path.abspath('../')
package_path = os.path.abspath('../../')
spk2py_package_path = os.path.abspath('../../spk2py')

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, source_path)
sys.path.insert(0, docs_path)
sys.path.insert(0, package_path)
sys.path.insert(0, spk2py_package_path)
os.environ['PYTHONPATH'] = package_path + ':' + os.environ.get('PYTHONPATH', '')
os.environ['PYTHONPATH'] = spk2py_package_path + ':' + os.environ.get('PYTHONPATH', '')

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Minimum version, enforced by sphinx
needs_sphinx = '4.3'

project = 'spk2py'
copyright = '2023, Flynn OConnell'
author = 'Flynn OConnell'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_design',
    'sphinx.ext.githubpages'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = '.rst'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_context = {"default_mode": "dark"}
add_function_parentheses = False

# -----------------------------------------------------------------------------
# Autosummary/numpydoc
# -----------------------------------------------------------------------------

autosummary_generate = True
numpydoc_show_class_members = False
