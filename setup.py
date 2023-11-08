# setup.py -*- Python -*-
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2022 Advanced Micro Devices, Inc.

import os
import re
from setuptools import setup, find_packages
from setuptools.command.install import install


__pkg_name__ = 'rubicon'

verstrline = open(os.path.join(__pkg_name__, '__init__.py'), 'r').read()
vsre = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(vsre, verstrline, re.M)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in "{}/__init__.py".'.format(__pkg_name__))


package_name = "%s" % __pkg_name__


setup(
    name=package_name,
    version=__version__,
    packages=["nni"],
    package_dir={
        "nni": "./rubicon/tools/nni",
    },
    include_package_data=True,
    entry_points = {
        'console_scripts': [
            '{0} = {0}:main'.format(__pkg_name__)
        ]
    }
)
