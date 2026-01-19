#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
  author="Guochu Deng",
  author_email='gc.deng.ansto@gmail.com',
  classifiers=[
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
  description="import and plot TAS data, especially for Taipan and Sika",
  license="MIT license",
  platforms=["Windows", "Linux", "Mac OS X", "Unix"],
  name='TasVisAn',
  version='0.1.2',
  zip_safe=False,
  packages=find_packages(include=['TasVisAn', 'TasVisAn.*']),
  install_requires=[
        'lmfit>=1.2.1',
        'pandas>=1.0.0',
        'numpy>=1.15.0',
        'scipy',
        'matplotlib>=3.1.0',
        'QtPy',
        'jupyter',
        'inspy',
        'plotly',
    ]
)