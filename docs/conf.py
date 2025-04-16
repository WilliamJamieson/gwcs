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
html_title = f"{project} v{release}"

# Output file base name for HTML help builder.
htmlhelp_basename = f"{project}doc"

# -- Options for LaTeX output --------------------------------------------------
latex_documents = [
    ("index", project + ".tex", project + " Documentation", author, "manual")
]
latex_logo = "_static/images/logo-light-mode.png"

# -- Options for manual page output --------------------------------------------
man_pages = [("index", project.lower(), project + " Documentation", [author], 1)]

# -- Add in additional sphinx extensions ----------------------------------------
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

# -- Options for html theme -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
# Override default settings from sphinx_asdf / sphinx_astropy (incompatible with furo)
html_sidebars = {}
# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/images/favicon.ico"
html_logo = ""

html_theme_options = {
    "light_logo": "images/logo-light-mode.png",
    "dark_logo": "images/logo-dark-mode.png",
}

pygments_style = "monokai"
# NB Dark style pygments is furo-specific at this time
pygments_dark_style = "monokai"
# Render inheritance diagrams in SVG
graphviz_output_format = "svg"

graphviz_dot_args = [
    "-Nfontsize=10",
    "-Nfontname=Helvetica Neue, Helvetica, Arial, sans-serif",
    "-Efontsize=10",
    "-Efontname=Helvetica Neue, Helvetica, Arial, sans-serif",
    "-Gbgcolor=white",
    "-Gfontsize=10",
    "-Gfontname=Helvetica Neue, Helvetica, Arial, sans-serif",
]
