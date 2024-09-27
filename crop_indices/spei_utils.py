# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 10:40:33 2022
These are some functions taken from the climate-indices package by monocongo to 
help to calculate the SPEI

@author: Florian Ellsäßer

Licence: This program is free software: you can redistribute it and/or modify 
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
#import numba
import numpy as np
import scipy
from enum import Enum
from math import exp, lgamma, pi, sqrt
# ------------------------------------------------------------------------------

## this is for the compute section

class Periodicity(Enum):
    """
    Enumeration type for specifying dataset periodicity.

    'monthly' indicates an array of monthly values, assumed to span full years,
    i.e. the first value corresponds to January of the initial year and any
    missing final months of the final year filled with NaN values,
    with size == # of years * 12

    'daily' indicates an array of full years of daily values with 366 days per year,
    as if each year were a leap year and any missing final months of the final
    year filled with NaN values, with array size == (# years * 366)
    """

    monthly = 12
    daily = 366

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return Periodicity[s]
        except KeyError:
            raise ValueError(f"No periodicity enumeration corresponding to {s}")

    def unit(self):
        if self.name == "monthly":
            unit = "month"
        elif self.name == "daily":
            unit = "day"
        else:
            raise ValueError(f"No periodicity unit corresponding to {self.name}")

        return unit
    
# ------------------------------------------------------------------------------
class Distribution(Enum):
    """
    Enumeration type for distribution fittings used for SPI and SPEI.
    """

    pearson = "pearson"
    gamma = "gamma"


# ------------------------------------------------------------------------------
# Retrieve logger and set desired logging level
#_logger = utils.get_logger(__name__, logging.DEBUG)

# ------------------------------------------------------------------------------
# valid upper and lower bounds for indices that are fitted/transformed to a distribution (SPI and SPEI)
_FITTED_INDEX_VALID_MIN = -3.09
_FITTED_INDEX_VALID_MAX = 3.09
    
    
## this is for the indices section 


# ------------------------------------------------------------------------------
#@numba.jit
def spei(
        precips_mm: np.ndarray,
        pet_mm: np.ndarray,
        scale: int,
        distribution: Distribution,
        periodicity: Periodicity,
        data_start_year: int,
        calibration_year_initial: int,
        calibration_year_final: int,
        fitting_params: dict = None,
) -> np.ndarray:
    """
    Compute SPEI fitted to the gamma distribution.

    PET values are subtracted from the precipitation values to come up with an array
    of (P - PET) values, which is then scaled to the specified months scale and
    finally fitted/transformed to SPEI values corresponding to the input
    precipitation time series.

    :param precips_mm: an array of monthly total precipitation values,
        in millimeters, should be of the same size (and shape?) as the input PET array
    :param pet_mm: an array of monthly PET values, in millimeters,
        should be of the same size (and shape?) as the input precipitation array
    :param scale: the number of months over which the values should be scaled
        before computing the indicator
    :param distribution: distribution type to be used for the internal
        fitting/transform computation
    :param periodicity: the periodicity of the time series represented by the
        input data, valid/supported values are 'monthly' and 'daily'
        'monthly' indicates an array of monthly values, assumed to span full
         years, i.e. the first value corresponds to January of the initial year
         and any missing final months of the final year filled with NaN values,
         with size == # of years * 12
         'daily' indicates an array of full years of daily values with 366 days
         per year, as if each year were a leap year and any missing final months
         of the final year filled with NaN values, with array size == (# years * 366)
    :param data_start_year: the initial year of the input datasets (assumes that
        the two inputs cover the same period)
    :param calibration_year_initial: initial year of the calibration period
    :param calibration_year_final: final year of the calibration period
    :param fitting_params: optional dictionary of pre-computed distribution
        fitting parameters, if the distribution is gamma then this dict should
        contain two arrays, keyed as "alphas" and "betas", and if the
        distribution is Pearson then this dict should contain four arrays keyed
        as "probabilities_of_zero", "locs", "scales", and "skews"
    :return: an array of SPEI values
    :rtype: numpy.ndarray of type float, of the same size and shape as the input
        PET and precipitation arrays
    """

    # if we're passed all missing values then we can't compute anything,
    # so we return the same array of missing values
    if (np.ma.is_masked(precips_mm) and precips_mm.mask.all()) \
            or np.all(np.isnan(precips_mm)):
        return precips_mm

    # validate that the two input arrays are compatible
    if precips_mm.size != pet_mm.size:
        message = "Incompatible precipitation and PET arrays"
       # _logger.error(message)
        raise ValueError(message)

    # clip any negative values to zero
    if np.amin(precips_mm) < 0.0:
     #   _logger.warn("Input contains negative values -- all negatives clipped to zero")
        precips_mm = np.clip(precips_mm, a_min=0.0, a_max=None)

    # subtract the PET from precipitation, adding an offset
    # to ensure that all values are positive
    p_minus_pet = (precips_mm.flatten() - pet_mm.flatten()) + 1000.0

    # remember the original length of the input array, in order to facilitate
    # returning an array of the same size
    original_length = precips_mm.size

    # get a sliding sums array, with each element's value
    # scaled by the specified number of time steps
    scaled_values = sum_to_scale(p_minus_pet, scale)

    if distribution is Distribution.gamma:

        # get (optional) fitting parameters if provided
        if fitting_params is not None:
            alphas = fitting_params["alphas"]
            betas = fitting_params["betas"]
        else:
            alphas = None
            betas = None

        # fit the scaled values to a gamma distribution and
        # transform to corresponding normalized sigmas
        transformed_fitted_values = \
            transform_fitted_gamma(
                scaled_values,
                data_start_year,
                calibration_year_initial,
                calibration_year_final,
                periodicity,
                alphas,
                betas,
            )

    elif distribution is Distribution.pearson:

        # get (optional) fitting parameters if provided
        if fitting_params is not None:
            probabilities_of_zero = fitting_params["probabilities_of_zero"]
            locs = fitting_params["locs"]
            scales = fitting_params["scales"]
            skews = fitting_params["skews"]
        else:
            probabilities_of_zero = None
            locs = None
            scales = None
            skews = None

        # fit the scaled values to a Pearson Type III distribution
        # and transform to corresponding normalized sigmas
        transformed_fitted_values = \
            transform_fitted_pearson(
                scaled_values,
                data_start_year,
                calibration_year_initial,
                calibration_year_final,
                periodicity,
                probabilities_of_zero,
                locs,
                scales,
                skews,
            )

    else:
        message = "Unsupported distribution argument: " + \
                  "{dist}".format(dist=distribution)
       # _logger.error(message)
        raise ValueError(message)

    # clip values to within the valid range, reshape the array back to 1-D
    values = \
        np.clip(transformed_fitted_values,
                _FITTED_INDEX_VALID_MIN,
                _FITTED_INDEX_VALID_MAX).flatten()

    # return the original size array
    return values[0:original_length]

#@numba.jit
def sum_to_scale(
        values: np.ndarray,
        scale: int,
) -> np.ndarray:
    """
    Compute a sliding sums array using 1-D convolution. The initial
    (scale - 1) elements of the result array will be padded with np.NaN values.
    Missing values are not ignored, i.e. if a np.NaN
    (missing) value is part of the group of values to be summed then the sum
    will be np.NaN

    For example if the first array is [3, 4, 6, 2, 1, 3, 5, 8, 5] and
    the number of values to sum is 3 then the resulting array
    will be [np.NaN, np.NaN, 13, 12, 9, 6, 9, 16, 18].

    More generally:

    Y = f(X, n)

    Y[i] == np.NaN, where i < n
    Y[i] == sum(X[i - n + 1:i + 1]), where i >= n - 1 and X[i - n + 1:i + 1]
        contains no NaN values
    Y[i] == np.NaN, where i >= n - 1 and X[i - n + 1:i + 1] contains
        one or more NaN values

    :param values: the array of values over which we'll compute sliding sums
    :param scale: the number of values for which each sliding summation will
        encompass, for example if this value is 3 then the first two elements of
        the output array will contain the pad value and the third element of the
        output array will contain the sum of the first three elements, and so on
    :return: an array of sliding sums, equal in length to the input values
        array, left padded with NaN values
    """

    # don't bother if the number of values to sum is 1
    if scale == 1:
        return values

    # get the valid sliding summations with 1D convolution
    sliding_sums = np.convolve(values, np.ones(scale), mode="valid")

    # pad the first (n - 1) elements of the array with NaN values
    return np.hstack(([np.NaN] * (scale - 1), sliding_sums))

    # BELOW FOR dask/xarray DataArray integration
    # # pad the values array with (scale - 1) NaNs
    # values = pad(values, pad_width=(scale - 1, 0), mode='constant', constant_values=np.NaN)
    #
    # start = 1
    # end = -(scale - 2)
    # return convolve(values, np.ones(scale), mode='reflect', cval=0.0, origin=0)[start: end]
    
#@numba.jit
def transform_fitted_gamma(
        values: np.ndarray,
        data_start_year: int,
        calibration_start_year: int,
        calibration_end_year: int,
        periodicity: Periodicity,
        alphas: np.ndarray = None,
        betas: np.ndarray = None,
) -> np.ndarray:
    """
    Fit values to a gamma distribution and transform the values to corresponding
    normalized sigmas.

    :param values: 2-D array of values, with each row typically representing a year
                   containing twelve columns representing the respective calendar
                   months, or 366 days per column as if all years were leap years
    :param data_start_year: the initial year of the input values array
    :param calibration_start_year: the initial year to use for the calibration period
    :param calibration_end_year: the final year to use for the calibration period
    :param periodicity: the type of time series represented by the input data,
        valid values are 'monthly' or 'daily'
        'monthly': array of monthly values, assumed to span full years,
        i.e. the first value corresponds to January of the initial year and any
        missing final months of the final year filled with NaN values, with
        size == # of years * 12
        'daily': array of full years of daily values with 366 days per year,
        as if each year were a leap year and any missing final months of the final
        year filled with NaN values, with array size == (# years * 366)
    :param alphas: pre-computed gamma fitting parameters
    :param betas: pre-computed gamma fitting parameters
    :return: 2-D array of transformed/fitted values, corresponding in size
        and shape of the input array
    :rtype: numpy.ndarray of floats
    """

    # if we're passed all missing values then we can't compute anything,
    # then we return the same array of missing values
    if (np.ma.is_masked(values) and values.mask.all()) or np.all(np.isnan(values)):
        return values

    # validate (and possibly reshape) the input array
    values = _validate_array(values, periodicity)

    # find the percentage of zero values for each time step
    zeros = (values == 0).sum(axis=0)
    probabilities_of_zero = zeros / values.shape[0]

    # replace zeros with NaNs
    values[values == 0] = np.NaN

    # compute fitting parameters if none were provided
    if (alphas is None) or (betas is None):
        alphas, betas = \
            gamma_parameters(
                values,
                data_start_year,
                calibration_start_year,
                calibration_end_year,
                periodicity,
            )

    # find the gamma probability values using the gamma CDF
    gamma_probabilities = scipy.stats.gamma.cdf(values, a=alphas, scale=betas)

    # TODO explain this better
    # (normalize including the probability of zero, putting into the range [0..1]?)
    probabilities = probabilities_of_zero + \
                    ((1 - probabilities_of_zero) * gamma_probabilities)

    # the values we'll return are the values at which the probabilities of
    # a normal distribution are less than or equal to the computed probabilities,
    # as determined by the normal distribution's quantile (or inverse
    # cumulative distribution) function
    return scipy.stats.norm.ppf(probabilities)


# ------------------------------------------------------------------------------
#@numba.jit
def _validate_array(
        values: np.ndarray,
        periodicity: Periodicity,
) -> np.ndarray:
    """

    :param values:
    :param periodicity:
    :return:
    """

    # validate (and possibly reshape) the input array
    if len(values.shape) == 1:

        if periodicity is None:
            message = "1-D input array requires a corresponding periodicity "\
                      "argument, none provided"
           # _logger.error(message)
            raise ValueError(message)

        elif periodicity is Periodicity.monthly:
            # we've been passed a 1-D array with shape (months),
            # reshape it to 2-D with shape (years, 12)
            values = reshape_to_2d(values, 12)

        elif periodicity is Periodicity.daily:
            # we've been passed a 1-D array with shape (days),
            # reshape it to 2-D with shape (years, 366)
            values = reshape_to_2d(values, 366)

        else:
            message = "Unsupported periodicity argument: '{0}'".format(periodicity)
          #  _logger.error(message)
            raise ValueError(message)

    elif (len(values.shape) != 2) or \
            ((values.shape[1] != 12) and (values.shape[1] != 366)):

        # neither a 1-D nor a 2-D array with valid shape was passed in
        message = "Invalid input array with shape: {0}".format(values.shape)
        #_logger.error(message)
        raise ValueError(message)

    return values

#@numba.jit
def reshape_to_2d(
        values: np.ndarray,
        second_axis_length: int,
) -> np.ndarray:
    """
    :param values: an 1-D numpy.ndarray of values
    :param second_axis_length:
    :return: the original values reshaped to 2-D, with shape
        [int(original length / second axis length), second axis length]
    :rtype: 2-D numpy.ndarray of floats
    """

    # if we've been passed a 2-D array with valid shape then let it pass through
    shape = values.shape
    if len(shape) == 2:
        if shape[1] == second_axis_length:
            # data is already in the shape we want, return it unaltered
            return values
        else:
            message = "Values array has an invalid shape (2-D but second " + \
                      "dimension not {dim}".format(dim=second_axis_length) + \
                      "): {shape}".format(shape=shape)
           # _logger.error(message)
            raise ValueError(message)

    # otherwise make sure that we've been passed a flat (1-D) array of values
    elif len(shape) != 1:
        message = "Values array has an invalid shape (not 1-D " + \
                  "or 2-D): {shape}".format(shape=shape)
        #_logger.error(message)
        raise ValueError(message)

    # pad the end of the original array in order
    # to have an ordinal increment, if necessary
    final_year_values = shape[0] % second_axis_length
    if final_year_values > 0:
        pads = second_axis_length - final_year_values
        values = np.pad(values,
                        pad_width=(0, pads),
                        mode="constant",
                        constant_values=np.NaN)

    # we should have an ordinal number of years now
    # (ordinally divisible by second_axis_length)
    first_axis_length = int(values.shape[0] / second_axis_length)

    # return the reshaped array
    return np.reshape(values, newshape=(first_axis_length, second_axis_length))

#@numba.jit
def transform_fitted_pearson(
        values: np.ndarray,
        data_start_year: int,
        calibration_start_year: int,
        calibration_end_year: int,
        periodicity: Periodicity,
        probabilities_of_zero: np.ndarray = None,
        locs: np.ndarray = None,
        scales: np.ndarray = None,
        skews: np.ndarray = None,
) -> np.ndarray:
    """
    Fit values to a Pearson Type III distribution and transform the values
    to corresponding normalized sigmas.

    :param values: 2-D array of values, with each row representing a year containing
                   twelve columns representing the respective calendar months,
                   or 366 columns representing days as if all years were leap years
    :param data_start_year: the initial year of the input values array
    :param calibration_start_year: the initial year to use for the calibration period
    :param calibration_end_year: the final year to use for the calibration period
    :param periodicity: the periodicity of the time series represented by the input
                        data, valid/supported values are 'monthly' and 'daily'
                        'monthly' indicates an array of monthly values, assumed
                        to span full years, i.e. the first value corresponds
                        to January of the initial year and any missing final
                        months of the final year filled with NaN values,
                        with size == # of years * 12
                        'daily' indicates an array of full years of daily values
                        with 366 days per year, as if each year were a leap year
                        and any missing final months of the final year filled
                        with NaN values, with array size == (# years * 366)
    :param probabilities_of_zero: pre-computed probabilities of zero for each
        month or day of the year
    :param locs: pre-computed loc values for each month or day of the year
    :param scales: pre-computed scale values for each month or day of the year
    :param skews: pre-computed skew values for each month or day of the year
    :return: 2-D array of transformed/fitted values, corresponding in size
             and shape of the input array
    :rtype: numpy.ndarray of floats
    """

    # sanity check for the fitting parameters arguments
    pearson_param_args = [probabilities_of_zero, locs, scales, skews]
    if any(param_arg is None for param_arg in pearson_param_args):
        if pearson_param_args.count(None) < len(pearson_param_args):
            raise ValueError(
                "At least one but not all of the Pearson Type III fitting "
                "parameters are specified -- either none or all of "
                "these must be specified")

    # if we're passed all missing values then we can't compute anything,
    # and we'll return the same array of missing values
    if (np.ma.is_masked(values) and values.mask.all()) or np.all(np.isnan(values)):
        return values

    # validate (and possibly reshape) the input array
    values = _validate_array(values, periodicity)

    # compute the Pearson Type III fitting values if none were provided
    if any(param_arg is None for param_arg in pearson_param_args):

        # determine the end year of the values array
        data_end_year = data_start_year + values.shape[0]

        # make sure that we have data within the full calibration period,
        # otherwise use the full period of record
        if (calibration_start_year < data_start_year) \
                or (calibration_end_year > data_end_year):
            calibration_start_year = data_start_year
            calibration_end_year = data_end_year

        # compute the values we'll use to fit to the Pearson Type III distribution
        probabilities_of_zero, locs, scales, skews = \
            pearson_parameters(
                values,
                data_start_year,
                calibration_start_year,
                calibration_end_year,
                periodicity,
            )

    # fit each value to the Pearson Type III distribution
    values = _pearson_fit(values, probabilities_of_zero, skews, locs, scales)

    return values


# ------------------------------------------------------------------------------
#@numba.jit
def gamma_parameters(
        values: np.ndarray,
        data_start_year: int,
        calibration_start_year: int,
        calibration_end_year: int,
        periodicity: Periodicity,
) -> (np.ndarray, np.ndarray):
    """
    Computes the gamma distribution parameters alpha and beta.

    :param values: 2-D array of values, with each row typically representing a year
                   containing twelve columns representing the respective calendar
                   months, or 366 days per column as if all years were leap years
    :param data_start_year: the initial year of the input values array
    :param calibration_start_year: the initial year to use for the calibration period
    :param calibration_end_year: the final year to use for the calibration period
    :param periodicity: the type of time series represented by the input data,
        valid values are 'monthly' or 'daily'
        'monthly': array of monthly values, assumed to span full years,
        i.e. the first value corresponds to January of the initial year and any
        missing final months of the final year filled with NaN values, with
        size == # of years * 12
        'daily': array of full years of daily values with 366 days per year,
        as if each year were a leap year and any missing final months of the final
        year filled with NaN values, with array size == (# years * 366)
    :return: two 2-D arrays of gamma fitting parameter values, corresponding in size
        and shape of the input array
    :rtype: tuple of two 2-D numpy.ndarrays of floats, alphas and betas
    """

    # if we're passed all missing values then we can't compute anything,
    # then we return an array of missing values
    if (np.ma.is_masked(values) and values.mask.all()) or np.all(np.isnan(values)):
        if periodicity is Periodicity.monthly:
            shape = (12,)
        elif periodicity is Periodicity.daily:
            shape = (366,)
        else:
            raise ValueError("Unsupported periodicity: {periodicity}".format(periodicity=periodicity))
        alphas = np.full(shape=shape, fill_value=np.NaN)
        betas = np.full(shape=shape, fill_value=np.NaN)
        return alphas, betas

    # validate (and possibly reshape) the input array
    values = _validate_array(values, periodicity)

    # replace zeros with NaNs
    values[values == 0] = np.NaN

    # determine the end year of the values array
    data_end_year = data_start_year + values.shape[0]

    # make sure that we have data within the full calibration period,
    # otherwise use the full period of record
    if (calibration_start_year < data_start_year) or \
            (calibration_end_year > data_end_year):
        calibration_start_year = data_start_year
        calibration_end_year = data_end_year

    # get the year axis indices corresponding to
    # the calibration start and end years
    calibration_begin_index = calibration_start_year - data_start_year
    calibration_end_index = (calibration_end_year - data_start_year) + 1

    # get the values for the current calendar time step
    # that fall within the calibration years period
    calibration_values = values[calibration_begin_index:calibration_end_index, :]

    # compute the gamma distribution's shape and scale parameters, alpha and beta
    # TODO explain this better
    means = np.nanmean(calibration_values, axis=0)
    log_means = np.log(means)
    logs = np.log(calibration_values)
    mean_logs = np.nanmean(logs, axis=0)
    a = log_means - mean_logs
    alphas = (1 + np.sqrt(1 + 4 * a / 3)) / (4 * a)
    betas = means / alphas

    return alphas, betas

# ------------------------------------------------------------------------------
#@numba.jit
def pearson_parameters(
        values: np.ndarray,
        data_start_year: int,
        calibration_start_year: int,
        calibration_end_year: int,
        periodicity: Periodicity,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    This function computes the probability of zero and Pearson Type III
    distribution parameters corresponding to an array of values.

    :param values: 2-D array of values, with each row representing a year
        containing either 12 values corresponding to the calendar months of
        that year, or 366 values corresponding to the days of the year
        (with Feb. 29th being an average of the Feb. 28th and Mar. 1st values for
        non-leap years) and assuming that the first value of the array is
        January of the initial year for an input array of monthly values or
        Jan. 1st of initial year for an input array daily values
    :param periodicity: monthly or daily
    :return: four 1-D array of fitting values for the Pearson Type III
        distribution, with shape (12,) for monthly or (366,) for daily

        returned array 1: probability of zero
        returned array 2: first Pearson Type III distribution parameter (loc)
        returned array 3 :second Pearson Type III distribution parameter (scale)
        returned array 4: third Pearson Type III distribution parameter (skew)
    """

    # reshape precipitation values to (years, 12) for monthly,
    # or to (years, 366) for daily
    if periodicity is Periodicity.monthly:

        values = reshape_to_2d(values, 12)

    elif periodicity is Periodicity.daily:

        values = reshape_to_2d(values, 366)

    else:

        raise ValueError("Invalid periodicity argument: %s" % periodicity)

    # validate that the values array has shape: (years, 12) for monthly or (years, 366) for daily
    if len(values.shape) != 2:
        message = "Invalid shape of input data array: {shape}".format(shape=values.shape)
        #_logger.error(message)
        raise ValueError(message)

    else:

        time_steps_per_year = values.shape[1]
        if (time_steps_per_year != 12) and (time_steps_per_year != 366):
            message = "Invalid shape of input data array: {shape}".format(shape=values.shape)
            #_logger.error(message)
            raise ValueError(message)

    # determine the end year of the values array
    data_end_year = data_start_year + values.shape[0]

    # make sure that we have data within the full calibration period,
    # otherwise use the full period of record
    if (calibration_start_year < data_start_year) or \
            (calibration_end_year > data_end_year):
        calibration_start_year = data_start_year
        calibration_end_year = data_end_year

    # get the year axis indices corresponding to
    # the calibration start and end years
    calibration_begin_index = calibration_start_year - data_start_year
    calibration_end_index = (calibration_end_year - data_start_year) + 1

    # get the values for the current calendar time step
    # that fall within the calibration years period
    calibration_values = values[calibration_begin_index:calibration_end_index, :]

    # the values we'll compute and return
    probabilities_of_zero = np.zeros((time_steps_per_year,))
    locs = np.zeros((time_steps_per_year,))
    scales = np.zeros((time_steps_per_year,))
    skews = np.zeros((time_steps_per_year,))

    # compute the probability of zero and Pearson
    # parameters for each calendar time step
    # TODO vectorize the below loop? create a @numba.vectorize() ufunc
    #  for application over the second axis
    for time_step_index in range(time_steps_per_year):

        # get the values for the current calendar time step
        time_step_values = calibration_values[:, time_step_index]

        # count the number of zeros and valid (non-missing/non-NaN) values
        number_of_zeros, number_of_non_missing = \
            count_zeros_and_non_missings(time_step_values)

        # make sure we have at least four values that are both non-missing (i.e. non-NaN)
        # and non-zero, otherwise use the entire period of record
        if (number_of_non_missing - number_of_zeros) < 4:

            # we can't proceed, bail out using zeros
            continue

        # calculate the probability of zero for the calendar time step
        probability_of_zero = 0.0
        if number_of_zeros > 0:

            probability_of_zero = number_of_zeros / number_of_non_missing

        # get the estimated L-moments, if we have
        # more than three non-missing/non-zero values
        if (number_of_non_missing - number_of_zeros) > 3:

            # get the Pearson Type III parameters for this time
            # step's values within the calibration period
            params = fit(time_step_values)
            probabilities_of_zero[time_step_index] = probability_of_zero
            locs[time_step_index] = params["loc"]
            scales[time_step_index] = params["scale"]
            skews[time_step_index] = params["skew"]

    return probabilities_of_zero, locs, scales, skews

# ------------------------------------------------------------------------------
def count_zeros_and_non_missings(
        values: np.ndarray,
) -> (int, int):
    """
    Given an input array of values return a count of the zeros
    and non-missing values. Missing values assumed to be numpy.NaNs.

    :param values: array like object (numpy array, most likely)
    :return: two int scalars: 1) the count of zeros, and
        2) the count of non-missing values
    """

    # make sure we have a numpy array
    values = np.array(values)

    # count the number of zeros and non-missing (non-NaN) values
    zeros = values.size - np.count_nonzero(values)
    non_missings = np.count_nonzero(~np.isnan(values))

    return zeros, non_missings

# ------------------------------------------------------------------------------
#@numba.jit
def _pearson_fit(
        values: np.ndarray,
        probabilities_of_zero: np.ndarray,
        skew: np.ndarray,
        loc: np.ndarray,
        scale: np.ndarray,
) -> np.ndarray:
    """
    Perform fitting of an array of values to a Pearson Type III distribution
    as described by the Pearson Type III parameters and probability of zero arguments.

    :param values: an array of values to fit to the Pearson Type III
        distribution described by the skew, loc, and scale
    :param probabilities_of_zero: probability that the value is zero
    :param skew: first Pearson Type III parameter, the skew of the distribution
    :param loc: second Pearson Type III parameter, the loc of the distribution
    :param scale: third Pearson Type III parameter, the scale of the distribution
    """

    # only fit to the distribution if the values array is valid/not missing
    if not np.all(np.isnan(values)):

        minimums_possible = _minimum_possible(skew, loc, scale)
        minimums_mask = values <= minimums_possible
        zero_mask = np.logical_and((values < 0.0005), (probabilities_of_zero > 0.0))
        trace_mask = np.logical_and((values < 0.0005), (probabilities_of_zero <= 0.0))

        # get the Pearson Type III cumulative density function value
        values = scipy.stats.pearson3.cdf(values, skew, loc, scale)

        # turn zero, trace, or minimum values either into either zero
        # or minimum value based on the probability of zero
        values[zero_mask] = 0.0
        values[trace_mask] = 0.0005

        # compute the minimum value possible, and if any values are below
        # that threshold then we set the corresponding CDF to a floor value
        # TODO ask Richard Heim why the use of this floor value, matching
        #  that used for the trace amount?
        nans_mask = np.isnan(values)
        values[np.logical_and(minimums_mask, nans_mask)] = 0.0005

        # account for negative skew
        skew_mask = skew < 0.0
        values[:, skew_mask] = 1 - values[:, skew_mask]

        if not np.all(np.isnan(values)):

            # calculate the probability value, clipped between 0 and 1
            probabilities = np.clip(
                (probabilities_of_zero + ((1.0 - probabilities_of_zero) * values)),
                0.0,
                1.0,
            )

            # the values we'll return are the values at which the probabilities
            # of a normal distribution are less than or equal to the computed
            # probabilities, as determined by the normal distribution's
            # quantile (or inverse cumulative distribution) function
            fitted_values = scipy.stats.norm.ppf(probabilities)

        else:

            fitted_values = values

    else:

        fitted_values = values

    return fitted_values

# ------------------------------------------------------------------------------
#@numba.jit
def _minimum_possible(
        skew: float,
        loc: float,
        scale: float,
) -> float:
    """
    Compute the minimum possible value that can be fitted to a distribution
    described by a set of skew, loc, and scale parameters.

    :param skew:
    :param loc:
    :param scale:
    :return:
    """

    alpha = 4.0 / (skew * skew)

    # calculate the lowest possible value that will
    # fit the distribution (i.e. Z = 0)
    return loc - ((alpha * scale * skew) / 2.0)

# ------------------------------------------------------------------------------
def fit(timeseries: np.ndarray) -> dict:
    """
    Returns the L-Moments fit (loc, scale, skew) corresponding to the
    input array of values.

    :param timeseries:
    :return:
    """

    # estimate the L-moments of the values array
    lmoments = _estimate_lmoments(timeseries)

    # validate the L-Moments
    if (lmoments[1] <= 0.0) or (abs(lmoments[2]) >= 1.0):
        message = "Unable to calculate Pearson Type III parameters " + \
                  "due to invalid L-moments"
        #_logger.error(message)
        raise ValueError(message)

    return _estimate_pearson3_parameters(lmoments)


# ------------------------------------------------------------------------------
# @numba.jit
def _estimate_lmoments(values: np.ndarray) -> np.ndarray:
    """
    Estimate sample L-moments, based on Fortran code written for inclusion
    in IBM Research Report RC20525,
    'FORTRAN ROUTINES FOR USE WITH THE METHOD OF L-MOMENTS, VERSION 3'
    by J. R. M. Hosking, IBM Research Division,
    T. J. Watson Research Center, Yorktown Heights, NY 10598, Version 3 August 1996.

    Documentation on the original Fortran routines found here:
        https://rdrr.io/cran/nsRFA/man/HW.original.html

    This is a Python translation of the original Fortran subroutine SAMLMR()
    and which has been optimized for calculating only the first three L-moments.

    :param values: 1-D (flattened) array of float values
    :return: an estimate of the first three sample L-moments
    :rtype: 1-D numpy array of floats (the first three sample L-moments
        corresponding to the input values)
    """

    # we need to have at least four values in order
    # to make a sample L-moments estimation
    number_of_values = np.count_nonzero(~np.isnan(values))
    if number_of_values < 4:
        message = "Insufficient number of values to perform sample L-moments estimation"
        #_logger.warning(message)
        raise ValueError(message)

    # sort the values into ascending order
    values = np.sort(values)

    sums = np.zeros((3,))

    for i in range(1, number_of_values + 1):
        z = i
        term = values[i - 1]
        sums[0] = sums[0] + term
        for j in range(1, 3):
            z -= 1
            term = term * z
            sums[j] = sums[j] + term

    y = float(number_of_values)
    z = float(number_of_values)
    sums[0] = sums[0] / z
    for j in range(1, 3):
        y = y - 1.0
        z = z * y
        sums[j] = sums[j] / z

    k = 3
    p0 = -1.0
    for _ in range(2):
        ak = float(k)
        p0 = -p0
        p = p0
        temp = p * sums[0]
        for i in range(1, k):
            ai = i
            p = -p * (ak + ai - 1.0) * (ak - ai) / (ai * ai)
            temp = temp + (p * sums[i])
        sums[k - 1] = temp
        k = k - 1

    lmoments = np.zeros((3,))
    if sums[1] != 0:
        lmoments[0] = sums[0]
        lmoments[1] = sums[1]
        lmoments[2] = sums[2] / sums[1]

    return lmoments

# ------------------------------------------------------------------------------
def _estimate_pearson3_parameters(lmoments: np.ndarray) -> dict:
    """
    Estimate parameters via L-moments for the Pearson Type III distribution,
    based on Fortran code written for inclusion in IBM Research Report RC20525,
    'FORTRAN ROUTINES FOR USE WITH THE METHOD OF L-MOMENTS, VERSION 3'
    by J. R. M. Hosking, IBM Research Division, T. J. Watson Research Center,
    Yorktown Heights, NY 10598

    This is a Python translation of the original Fortran subroutine
    named 'pearson3'.

    :param lmoments: 3-element, 1-D (flat) array containing the first
        three L-moments (lambda-1, lambda-2, and tau-3)
    :return the Pearson Type III parameters corresponding to the input L-moments
    :rtype: a 3-element, 1-D (flat) numpy array of floats (loc, scale, skew)
    """

    c1 = 0.2906
    c2 = 0.1882
    c3 = 0.0442
    d1 = 0.36067
    d2 = -0.59567
    d3 = 0.25361
    d4 = -2.78861
    d5 = 2.56096
    d6 = -0.77045
    t3 = abs(lmoments[2])  # L-skewness?

    # ensure the validity of the L-moments
    if (lmoments[1] <= 0) or (t3 >= 1):
        message = "Unable to calculate Pearson Type III parameters due to invalid L-moments"
        #_logger.error(message)
        raise ValueError(message)

    # initialize the output values
    # loc, scale, skew

    # the first Pearson Type III parameter is the same as the first L-moment
    loc = lmoments[0]
    # pearson3_parameters = np.zeros((3,))

    # # the first Pearson Type III parameter is the same as the first L-moment
    # pearson3_parameters[0] = lmoments[0]

    if t3 <= 1e-6:
        # skewness is effectively zero
        scale = lmoments[1] * sqrt(pi)
        skew = 0.0
        # pearson3_parameters[1] = lmoments[1] * sqrt(pi)

    else:
        if t3 < 0.333333333:
            t = pi * 3 * t3 * t3
            alpha = (1.0 + (c1 * t)) / (t * (1.0 + (t * (c2 + (t * c3)))))
        else:
            t = 1.0 - t3
            alpha = t * (d1 + (t * (d2 + (t * d3)))) / (1.0 + (t * (d4 + (t * (d5 + (t * d6))))))

        alpha_root = sqrt(alpha)
        beta = sqrt(pi) * lmoments[1] * exp(lgamma(alpha) - lgamma(alpha + 0.5))
        scale = beta * alpha_root
        # pearson3_parameters[1] = beta * alpha_root

        # the sign of the third L-moment determines
        # the sign of the third Pearson Type III parameter
        if lmoments[2] < 0:
            skew = -2.0 / alpha_root
            # pearson3_parameters[2] = -2.0 / alpha_root
        else:
            skew = 2.0 / alpha_root
            # pearson3_parameters[2] = 2.0 / alpha_root

    # return pearson3_parameters
    return {"loc": loc, "skew": skew, "scale": scale}