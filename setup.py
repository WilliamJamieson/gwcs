#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import sys

from setuptools import setup, find_packages
from configparser import ConfigParser


conf = ConfigParser()
conf.read(['setup.cfg'])
metadata = dict(conf.items('metadata'))

PACKAGENAME = metadata.get('name', 'packagename')
# DESCRIPTION = metadata.get('description', 'Astropy affiliated package')
# AUTHOR = metadata.get('author', '')
# AUTHOR_EMAIL = metadata.get('author_email', '')
# LICENSE = metadata.get('license', 'unknown')
# URL = metadata.get('url', 'http://astropy.org')


def get_package_data():
    # Installs the schema files
    schemas = []
    root = os.path.join(PACKAGENAME, 'schemas')
    for node, dirs, files in os.walk(root):
        for fname in files:
            if fname.endswith('.yaml'):
                schemas.append(
                    os.path.relpath(os.path.join(node, fname), root))
    # In the package directory, install to the subdirectory 'schemas'
    schemas = [os.path.join('schemas', s) for s in schemas]
    return schemas


schemas = get_package_data()
PACKAGE_DATA ={'gwcs':schemas}


setup(use_scm_version=True,
      setup_requires=['setuptools_scm'],
      packages=find_packages(),
      package_data=PACKAGE_DATA,
)
