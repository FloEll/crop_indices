# -*- coding: utf-8 -*-
"""
First version created on Wed Jan 20 14:23:52 2021
@author: Florian Ellsäßer
Licence: CC BY-NC-SA https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

This script contains all the utile functions to support the calculation of the Heat 
Magnitude Day Index (HMD), the Standardized Precipitation Evapotranspiration Index 
(SPEI) and the Compound Stress Index (CSI). 
"""

import pandas as pd 
import numpy as np
import xarray as xr
from shapely.geometry import mapping
  
def format_natural_areas_raster_for_model_input(in_raster,in_shape,mask_params):
    """
        This function formats the natural area raster for model/index calculation
        input.
        
        Parameters:
            in_raster (xarray.core.dataset.Dataset) = raster with the natural areas
            in_shape (geopandas.geodataframe.GeoDataFrame) = shape file with the
            outlines of area of interest
            mask_params (list) = list with mask parameters
            
        Returns:
            out_raster (xarray.core.dataarray.DataArray) = raster with cut natural
            areas
    """
    # add a time that doesn't exist in t_max_germany_cropped
    in_raster.coords['time'] = np.array([1])
    # rename the variable
    in_raster = in_raster.rename({'__xarray_dataarray_variable__':'nat_areas'})
    # just get the nat_areas variable
    in_raster = in_raster.nat_areas
    # reduce the size of the array using the masking parameters
    in_raster = reduce_size_of_xarray(in_raster,mask_params)
    # now cut the values to the size of the shapefile
    in_raster = clip_xarray_to_shapefile(in_raster,'epsg:4326',in_shape)
    # remove all -9999
    in_raster = in_raster.where(in_raster != -9999.) 
    # drop the spatial ref variable
    out_raster = in_raster.drop('spatial_ref')
    
    return out_raster

# create a function to format the HMD input
def format_hmd_input(in_t_max_array,in_natural_areas_raster,in_shape,mask_params):
    """
        This function formats the hmd input. 
        
        Parameters:
            in_t_max_array (xarray.core.dataarray.DataArray) = Tmax data
            in_natural_areas_raster(xarray.core.dataset.Dataset) = natural areas data
            in_shape (geopandas.geodataframe.GeoDataFrame) = shapefile of area 
            of interst
            mask_params (list) = parameters to reduce size of netCDF/xarray 
            
        Returns:
            final_array (xarray.core.dataarray.DataArray) = input file for hmd
            index calculation            
    """
    
    # reformat the natural areas first 
    in_natural_areas_raster = format_natural_areas_raster_for_model_input(in_natural_areas_raster,
                                                                                in_shape,
                                                                                mask_params)
    # now we concatenate all three arrays
    final_array = xr.concat([in_t_max_array, 
                             #t_max_average_five_years,
                             in_natural_areas_raster], dim='time')    
    # rearrange (transpose) the final arrays dimensions
    final_array = final_array.transpose('time','longitude','latitude') 
    
    return final_array

# get a function for spei input
def format_spei_input(in_t_max,in_t_min,in_t_mean,in_precip,in_latitude,in_natural_area):
    """
        This function arranges the input for the calculation of the spei index.
        
        Parameters:
            in_t_max (xarray.core.dataarray.DataArray) = input data array Tmax
            in_t_min (xarray.core.dataarray.DataArray) = input data array Tmin
            in_t_mean (xarray.core.dataarray.DataArray) = input data array Tmean
            in_precip (xarray.core.dataarray.DataArray) = input data array precipitation
            in_latitude (xarray.core.dataarray.DataArray) = input data array for
            latitudes
            in_natural_area (xarray.core.dataarray.DataArray) = input data array
            for natural areas
            
        Returns:
            return concat_set_xarray (xarray.core.dataarray.DataArray) = output 
            array with concatenated data.
            
    """
    
    # create an array with the times 
    time_array = in_t_max.time
    # create the input for reconversion to xarray
    array_lat = in_t_max.latitude
    array_lon = in_t_max.longitude
    time_array = in_t_max.time
    array_time = np.arange(0,time_array.size*4 + 2)
    # get new index arrays 
    new_index_array_1 = np.arange(0,time_array.size) # index for t_max
    new_index_array_2 = np.arange(time_array.size,time_array.size*2) # index for t_min
    new_index_array_3 = np.arange(time_array.size*2,time_array.size*3) # index for t_mean
    new_index_array_4 = np.arange(time_array.size*3,time_array.size*4) # index for precip
    # set new index for the time dim
    in_t_max['time'] = new_index_array_1
    in_t_min['time'] = new_index_array_2
    in_t_mean['time'] = new_index_array_3
    in_precip['time'] = new_index_array_4
    # convert from xarray to numpy array
    in_t_max_numpy = np.array(in_t_max)
    in_t_min_numpy = np.array(in_t_min)
    in_t_mean_numpy = np.array(in_t_mean)
    in_precip_numpy = np.array(in_precip)
    in_latitude_numpy = np.array(in_latitude)
    in_natural_area_numpy = np.array(in_natural_area)
    # concatenate the data sets 
    concat_set_numpy = np.concatenate((in_t_min_numpy, in_t_max_numpy), axis=2)
    concat_set_numpy = np.concatenate((in_t_mean_numpy, concat_set_numpy), axis=2)
    concat_set_numpy = np.concatenate((in_precip_numpy, concat_set_numpy), axis=2)
    concat_set_numpy = np.concatenate((in_latitude_numpy, concat_set_numpy), axis=2)
    concat_set_numpy = np.concatenate((in_natural_area_numpy, concat_set_numpy), axis=2)
    # now reformat everything to xarray again
    concat_set_xarray = xr.DataArray(data=concat_set_numpy,
                  dims=['latitude', 'longitude','time'],
                  coords=dict(longitude=( array_lon),
                              latitude=( array_lat),
                              time=array_time),
                  attrs=dict(description='SPEI input data', ),
                                    )
    return concat_set_xarray

# get the latitude raster
def get_formatted_latitude_raster(in_natural_areas_raster,mask_params,in_shape):
    """
        This function formates the latitude for the input of the SPEI index. 
        
        Parameters:
            in_natural_areas_raster (xarray.core.dataset.Dataset= = a template
            data set for the latitudes
            mask_params (list) = list that contains the coordinates for cutting
            the ratser
            in_shape (geopandas.geodataframe.GeoDataFrame) = shapefile that contains
            the outlines of the area of interest
            
        Returns:
            latitude_array (xarray.core.dataarray.DataArray) = array with only 
            latitude values            
    """
    
    # call a function to just get the latitudes
    latitude_array = get_latitude_raster(in_natural_areas_raster,-9999.0)
    # create an xarray from this
    latitude_array = xr.DataArray(data = latitude_array, 
                 coords = {'latitude' : in_natural_areas_raster.latitude,
                           'longitude' : in_natural_areas_raster.longitude,
                           'time' : in_natural_areas_raster.time},

                 attrs = in_natural_areas_raster.attrs)
    # reduce the size using the masking parameters list
    latitude_array = reduce_size_of_xarray(latitude_array,mask_params)
    # clip it to the outlines of the shapefile
    latitude_array = clip_xarray_to_shapefile(latitude_array,'epsg:4326',in_shape)
    # remove the -9999 values
    latitude_array = latitude_array.where(latitude_array != -9999.) 
    # remove the saptial ref variable
    latitude_array = latitude_array.drop('spatial_ref')  
    return latitude_array

def format_csi_input(in_hmd,in_spei,in_yield):
    """
        This function arranges the input for the calculation of the CSI index
        
        Parameters:
            in_hmd (xarray.core.dataarray.DataArray) = hmd results array
            in_spei (xarray.core.dataarray.DataArray) = spei results array
            in_yield (xarray.core.dataarray.DataArray) = yield productivity array
            
        Returns:
            combined (xarray.core.dataarray.DataArray) = combined input file
            
    """
    
    # concat the arrays along he year dimension
    combined = xr.concat([in_hmd,in_spei,in_yield], dim='year')
    # fix the year coordinate
    combined = combined.assign_coords(year = np.array(np.arange(0,in_hmd.shape[2]*3)))
        
    return combined

def reformat_output(result_file, in_data_attributes, datetime_array):
    """
        This function reformats the output of the HMD and SPEI index calculation.
        
        Parameters:
            result_file (xarray.core.dataarray.DataArray) = index results file
            in_data_attributes (dict) = dictionary with the desired output array attributes
            datetime_array (numpy.ndarray) = containing the years as numpy datetime
            
        Returns:
            reformatted_result_file (xarray.core.dataarray.DataArray) = reformatted
            output xarray            
    """
    
    # create a test set where spatial_ref is the years 
    try:
        reformatted_result_file = result_file.tx.assign_coords(spatial_ref = result_file.tx.year)
    except:
        reformatted_result_file = result_file.assign_coords(spatial_ref = result_file.year)
    # add the in data attributes in_data_attributes
    reformatted_result_file.attrs = in_data_attributes
    # create a test set where spatial_ref is the years 
    reformatted_result_file = reformatted_result_file.assign_coords(years = reformatted_result_file.year)
    # transpose the dimensions
    reformatted_result_file = reformatted_result_file.transpose('latitude','longitude','year')
    # drop the spatial ref variable
    reformatted_result_file = reformatted_result_file.reset_coords(names = ['spatial_ref'], drop=True)
    # now remove years variable
    reformatted_result_file = reformatted_result_file.reset_coords(names = ['years'], drop=True)
    # first get the years (as the first of every year for now)
    dates = pd.DatetimeIndex(datetime_array)
    # now get all the unique years
    dates = dates.year.unique()
    # format from datetime to year
    dates = pd.to_datetime(dates, format='%Y')
    # now change coordinates to datetime objects
    reformatted_result_file = reformatted_result_file.assign_coords({'year': (dates)})
    
    return reformatted_result_file

def reformat_output_csi(result_file, in_data_attributes, datetime_array):
    """
        This function reformats the output of the CSI index calculation 
        
        Parameters:
            result_file (xarray.core.dataarray.DataArray) = index results file
            in_data_attributes (dict) = dictionary with the desired output array attributes
            datetime_array (numpy.ndarray) = containing the years as numpy datetime
            
        Returns:
            reformatted_result_file (xarray.core.dataarray.DataArray) = reformatted
            output xarray              
    """
    
    # define the spatial_ref as the years     
    try:
        reformatted_result_file = result_file.tx.assign_coords(spatial_ref = result_file.tx.year)
    except:
        reformatted_result_file = result_file.assign_coords(spatial_ref = result_file.year)
    # add the in data attributes in_data_attributes
    reformatted_result_file.attrs = in_data_attributes
    # create a test set where spatial_ref is the years 
    reformatted_result_file = reformatted_result_file.assign_coords(years = reformatted_result_file.year)
    # transpose the dimensions
    reformatted_result_file = reformatted_result_file.transpose('year','latitude','longitude')
    # drop the spatial ref
    reformatted_result_file = reformatted_result_file.reset_coords(names = ['spatial_ref'], drop=True)
    # now remove years 
    reformatted_result_file = reformatted_result_file.reset_coords(names = ['years'], drop=True)
    # get the dates
    dates = datetime_array
    # now change the coordinates to datetime objects
    reformatted_result_file = reformatted_result_file.assign_coords({'year': (dates)})
    
    return reformatted_result_file

# cut the yield data to the shape of the other xarray
def format_yield_data(in_yield_data,mask_params,in_shape):
    """
        This function reformats the yield data.
        
        Parameters:
            in_yield_data (xarray.core.dataarray.DataArray) = data array with y
            mask_params (list) = list with masking parameters
            in_shape (geopandas.geodataframe.GeoDataFrame) = shapefile with outlines
            of area of interest
            
        Returns:
            yield_data (xarray.core.dataarray.DataArray) = formatted yield data
    """
    # cut array to coordinates defined in mask params list
    yield_data = reduce_size_of_xarray(in_yield_data,mask_params)
    # cut to area of interest
    yield_data = clip_xarray_to_shapefile(yield_data,'epsg:4326',in_shape)
    # remove the -9999 values
    yield_data = yield_data.where(yield_data != -9999.) 
    # drop the spatial_ref variable
    yield_data = yield_data.drop('spatial_ref')
    
    return yield_data
              
def getCoordinates(in_data):
    """
        This function takes the Tmax E-Obs data and returns three arrays with 
        the dimensions (longitude, latitude and time)
        
        Parameters:
            in_data (xarray.core.dataarray.DataArray) = input data such as Tmax
            
        Returns:
            longitude_array (numpy.ndarray) = array with rounded coordinates
            latitude_array (numpy.ndarray) = array with rounded coordinates
            time_array (numpy.ndarray) = array with times            
    """
    
    # get the longitude and round values to two digits               
    longitude_array = np.round(in_data.longitude.values,2)
    # get the latitude and round values to two digits
    latitude_array = np.round(in_data.latitude.values,2)
    # get the time as an array
    time_array = in_data.time.values
        
    return longitude_array, latitude_array, time_array

def get_latitude_raster(in_data_set,nan_value):
    """
        This function creates an array with the latitudes of only the input array.
        This is useful for the spei index calculation.
        
        Parameters:
            in_data_set (xarray.core.dataarray.DataArray) = Template array, in 
            this case it is the natural areas raster
            nan_value (float) = Nan value, usually -9999.0
            
        Returns:
            latitude_array (xarray.core.dataarray.DataArray) = array with each 
            value being the associated latitude            
    """
    
    # access the variable array
    in_array = in_data_set.__xarray_dataarray_variable__
    # get the shape of the array
    latitude_array_shape = in_array.shape
    # create an empty array in the shape of the in_array
    latitude_array = np.zeros(latitude_array_shape)
    # now fill the empty array with the latitude values
    for counter_longitude, row in enumerate(in_array):
        # do this row by row
        for counter_latitude, item in enumerate(row):
            # exclude the nan_values
            if item.values != nan_value:
                # add the latitude values
                latitude_array[counter_longitude,counter_latitude] = item.latitude.values
                
    return latitude_array 


def round_coordinates(in_array):
    """
        This function rounds the coordinates two two decimals. This prevents errors
        due to weird roundings that sometimes appear
        
        Parameters:
            in_array (xarray.core.dataarray.DataArray) = input Data array e.g. Tmax
            
        Returns:
            in_array (xarray.core.dataarray.DataArray) = array with rounded coordinates
    """
    
    # round longitude
    in_array.coords['longitude'] = np.round(in_array.coords['longitude'],2)
    # round latitude    
    in_array.coords['latitude'] = np.round(in_array.coords['latitude'],2)
    
    return in_array
# 
def reformat_data_array(in_array):
    """
        This function drops the spatial_ref dimension/coord.
        
        Parameters:
            in_array (xarray.core.dataarray.DataArray) = input Data array e.g. Tmax
            
        Returns:
            in_array (xarray.core.dataarray.DataArray) = array with rounded coordinates            
    """
    
    # drop the spatial ref
    in_array = in_array.reset_coords(names = ['spatial_ref'], drop=True)
    
    return in_array

def reduce_size_of_xarray(in_array,mask_params):
    """
        This function reduces the size of the xarray. 
        
        Parameters:
            in_array (xarray.core.dataarray.DataArray) = input Data array e.g. Tmax
            mask_params (list) = list that contains the coordinate parameters for masking
            
        Returns:
            out_array(xarray.core.dataarray.DataArray) = masked array            
    """
    
    # unpack the mask params 
    min_lon, min_lat, max_lon, max_lat, min_time, max_time = mask_params
    # create a mask
    mask_lon = (in_array.longitude >= min_lon) & (in_array.longitude <= max_lon)
    mask_lat = (in_array.latitude >= min_lat) & (in_array.latitude <= max_lat)
    # cut to mask 
    in_array = in_array.rio.write_crs('epsg:4326', inplace=True)
    # drop all the rest outside of the mask
    out_array = in_array.where(mask_lon & mask_lat, drop=True)
    
    return out_array

# create a function to clip the xarray to a Germany shape
def clip_xarray_to_shapefile(in_array,in_crs,in_shape):
    """
        This function clips the xarray to a shape file
        
        Parameters:
            in_array (xarray.core.dataarray.DataArray) = input Data array e.g. Tmax
            in_crs (str) = with definition of coordinate system e.g. 'epsg:4326'
            in_shape (geopandas.geodataframe.GeoDataFrame) = shapefile with outlines
            of area of interest
            
        Returns:
            out_array (xarray.core.dataarray.DataArray) = output data that has been 
            cut to size            
    """
    
    # write the CRS to the raster data
    in_array.rio.write_crs(in_crs, inplace=True)
    # cut E-OBS rasters to the germany_shape shapefile 
    out_array = in_array.astype('float32').rio.clip(in_shape.geometry.apply(mapping), in_shape.crs, drop=True)

    return out_array

def get_mean_value_array(in_array):
    """
        This function calculate the average for every day of the year. 
        
        Parameters:
            in_array (numpy.ndarray) = numpy array with input data
            
        Returns:
            out_array (numpy.ndarray) = numpy array with output data
            
    """
    
    # check if only nan, then skip
    if np.all(np.isnan(in_array)):
        # retrun an empty array
        return np.array([np.nan]*365)
    else:
        # get the number of in years
        total_years = round(len(in_array)/365)
        # create an empty pandas data frame and loop through it
        df = pd.DataFrame()
        #loop through the years
        for count, year in enumerate(np.arange(0,total_years)):
            new_array = in_array[count*365:(count+1)*365]
            # now add the new array to the pandas dataframe as a column
            name = 'in_data_year_' + str(count)
            df[name] = new_array
        # now get the mean of all years 
        df['mean'] = df.mean(axis=1)
        
        return np.array(df['mean'].values)

def get_mean_array(in_array):
    """
        This function the mean of an array and returns a 2D raster
        
        Parameters:
            in_array (xarray.core.dataarray.DataArray) = input Data array e.g. Tmax
            
        Returns:
            mean_array (xarray.core.dataarray.DataArray) = 2D array with the means            
    """
    # apply ufunc and call the get_mean_value_array function
    mean_array = xr.apply_ufunc(get_mean_value_array, 
                         in_array,
                         input_core_dims=[['time']],
                         output_core_dims=[['time_average']], 
                         dask = 'parallelized', 
                         vectorize = True)
    
    return mean_array