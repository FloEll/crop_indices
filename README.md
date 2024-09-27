# crop_indices

This package is still under construction and the scientific background study still under review! It is work in progress and might change significantly. 

## What is it?

**crop_indices** is a Python package to calculate the HMD (Heat Magnitude Day), SPEI (Standardized Precipitation-Evapotranspiration Index) and CSI (Combined Stress Index) indices to estimate weather impact on crops

## Main features

This package contains all the essential methods for the Heat Magnitude Day Index 
(HMD) and the Combined Stress Index (CSI). It further contains the crop-adaption
for the Standardized Precipitation Evapotranspiration Index (SPEI) however, the 
actual SPEI code is taken from the climate_indices package. 

## Where to get it

The source code is currently hosted on GitHub at: https://github.com/FloEll/crop_indices

It can be installed using pip:

`pip install crop_indices`

## Dependencies

The [NumPy](https://pypi.org/project/numpy/), [pandas](https://pypi.org/project/pandas/), [xarray](https://pypi.org/project/xarray/), [shapely](https://pypi.org/project/shapely/), [DateTime](https://pypi.org/project/DateTime/), [scikit-learn](https://pypi.org/project/scikit-learn/) packages are (if not already available) installed automatically when installing crops.

When working with netCDF data it is also useful to install the [netcdf4](https://pypi.org/project/netCDF4/) package. 

## Licence

GNU General Public License 3, see https://www.gnu.org/licenses/.

## Background

The work on **crop_indices** started in the [CROP project](https://www.uni-giessen.de/fbz/zentren/zeu/activities/researchprojects/CROP) at Justus-Liebig-University of Gießen in 2022. 

## Discussion and Development

If you are missing a feature or an index, did you find a bug or an error in the code or do you just have a great idea of how to improve this package, please open an issue at https://github.com/FloEll/crop_indices/issues

## How to cite

We are currently working on a manuscript that will also cover this package. Until then you can use the following citation: Florian Ellsäßer, 2024, crops python package to analyze harvest anomalies, https://pypi.org/project/crop_indices/ 
