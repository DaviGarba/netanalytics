#!/usr/bin/python
"""netanalytics setup script.

Author: Veronica Tozzo
Copyright (c) 2018, Veronica Tozzo.
Licensed under the BSD 3-Clause License (see LICENSE.txt).
"""

from setuptools import setup, find_packages
# import numpy as np

setup(
    name='netanalytics',

    description=('netanalytics'),
    long_description=open('README.md').read(),
    author='Veronica Tozzo',
    author_email='veronica.tozzo@dibris.unige.it',
    maintainer='Veronica Tozzo',
    maintainer_email='veronica.tozzo@dibris.unige.it',
    keywords=['graphs', 'analysis'],
    classifiers=[
        'Development Status :: 1 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'License :: OSI Approved :: BSD License',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Natural Language :: English',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python'
    ],
    license='FreeBSD',
    packages=find_packages(exclude=["*.__old", "*.tests"]),
    include_package_data=True,
    requires=['numpy (>=1.11)',
              'scipy (>=0.16.1,>=1.0)',
              'sklearn (>=0.17)',
              'six'],
)
