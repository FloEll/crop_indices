# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:21:24 2024

@author: Florian Ellsäßer
"""

import setuptools

setuptools.setup(name='crop_indices',
      version='0.0.1',
      description='Calculate the HMD (Heat Magnitude Day), SPEI (Standardized Precipitation-Evapotranspiration Index) and CSI (Combined Stress Index) indices to estimate weather impact on crops.',
      author='Florian Ellsäßer',
      author_email='f.j.ellsaesser@utwente.nl',
      url='',
      project_urls={
        'Documentation': 'https://github.com/FloEll/crop_indices',
        'Source': 'https://github.com/FloEll/crop_indices',
        'Tracker': 'https://github.com/FloEll/crop_indices/issues',
      },
      license='GPL-3.0',
      packages=['crop_indices'],
      install_requires=[
          'numpy','pandas','xarray', 'shapely', 'datetime', 'scikit-learn'
      ]
      )