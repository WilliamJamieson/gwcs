# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Astropy documentation build configuration file.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this file.
#
# All configuration values have a default. Some values are defined in
# the global Astropy configuration which is loaded here before anything else.
# See astropy.sphinx.conf for which values are set there.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('..'))
# IMPORTANT: the above commented section was generated by sphinx-quickstart, but
# is *NOT* appropriate for astropy or Astropy affiliated packages. It is left
# commented out with this explanation to make it clear why this should not be
# done. If the sys.path entry above is added, when the astropy.sphinx.conf
# import occurs, it will import the *source* version of astropy instead of the
# version installed (if invoked as "make html" or directly with sphinx), or the
# version in the build directory (if "python setup.py build_sphinx" is used).
# Thus, any C-extensions that are needed to build the documentation will *not*
# be accessible, and the documentation will not build correctly.

import sys
import tomllib
from datetime import datetime
from pathlib import Path

if sys.version_info >= (3, 12):
    from importlib.metadata import distribution
else:
    from importlib_metadata import distribution

try:
    from sphinx_astropy.conf.v2 import *  # noqa: F403
except ImportError:
    print(  # noqa: T201
        "ERROR: the documentation requires the sphinx-astropy package to be installed"
    )
    sys.exit(1)

# -- General configuration ----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.2'

# To perform a Sphinx version check that needs to be more specific than
# major.minor, call `check_sphinx_version("x.y.z")` here.
# check_sphinx_version("1.2.1")

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns.append("_templates")  # noqa: F405

# This is added to the end of RST files - a good place to put substitutions to
# be used globally.
rst_epilog += """
"""  # noqa: F405

# Top-level directory containing ASDF schemas (relative to current directory)
asdf_schema_path = "../gwcs/schemas"
# This is the prefix common to all schema IDs in this repository
asdf_schema_standard_prefix = "stsci.edu/gwcs"
asdf_schema_reference_mappings = [
    (
        "tag:stsci.edu:asdf",
        "http://asdf-standard.readthedocs.io/en/latest/generated/stsci.edu/asdf/",
    ),
]

# -- Project information ------------------------------------------------------

# This does not *have* to match the package name, but typically does
with (Path(__file__).parent.parent / "pyproject.toml").open("rb") as metadata_file:
    configuration = tomllib.load(metadata_file)
    metadata = configuration["project"]
project = metadata["name"]
author = metadata["authors"][0]["name"]
copyright = f"{datetime.now().year}, {author}"  # noqa: A001

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

release = distribution(project).version
# for example take major/minor
version = ".".join(release.split(".")[:2])

# -- Options for HTML output ---------------------------------------------------

# A NOTE ON HTML THEMES
# The global astropy configuration uses a custom theme, 'bootstrap-astropy',
# which is installed along with astropy. A different theme can be used or
# the options for this theme can be modified by overriding some of the
# variables set in the global configuration. The variables set in the
# global configuration are listed below, commented out.

# Add any paths that contain custom themes here, relative to this directory.
# To use a different custom theme, add the directory containing the theme.
# html_theme_path = []

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes. To override the custom theme, set this to the
# name of a builtin theme or the name of a custom theme in html_theme_path.
# html_theme = None


# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = ''

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = ''

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = f"{project} v{release}"

# Output file base name for HTML help builder.
htmlhelp_basename = project + "doc"

# -- Options for LaTeX output --------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
# latex_documents = [('index', project + '.tex', project + u' Documentation',
#                    author, 'manual')]


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [("index", project.lower(), project + " Documentation", [author], 1)]

sys.path.insert(0, str(Path(__file__).parent / "sphinxext"))
extensions += ["sphinx_asdf"]  # noqa: F405

# Enable nitpicky mode - which ensures that all references in the docs resolve.
nitpicky = True
nitpick_ignore = [
    ("py:class", "gwcs.api.GWCSAPIMixin"),
    ("py:class", "gwcs.wcs._pipeline.Pipeline"),
    ("py:obj", "astropy.modeling.projections.projcodes"),
    ("py:attr", "gwcs.WCS.bounding_box"),
    ("py:meth", "gwcs.WCS.footprint"),
]
