"""
General collection of functions for calculating features of generic data, including various statistical tests and fitting functions.
"""

import numpy as np
import pandas as pd
from scipy import stats
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from GEN_Utils.LoggerConfig import logger_config

logger = logger_config(__name__)
logger.info('Import ok')

def row_mean(df, cols, result_name):
    """calculates mean of columns per row of a dataframe.
    Parameters
    ----------
    df : dataframe
        Description of parameter `df`.
    cols : list
        List of column names to be used for calculation
    result_name : list
        Name of column in which output is stored
    Returns
    -------
    dataframe
        Returns entire input dataframe, with appended mean columns
    """

    for i in range(df.shape[0]):
        vals = df.loc[i, cols]
        vals = vals.dropna()
        mean_val = np.mean(vals)
        df.loc[i, result_name] = mean_val
    return df


def mean_med_calc(vals, samplename):
    """Calculates the mean and median for a seet of values, appended to dictionary mapped to samplename"""
    vals = vals.dropna()
    mean_val = np.mean(vals)
    median_val = np.median(vals)
    calcs_dict[samplename] = [mean_val, median_val]
    return calcs_dict


def filter_NaNs(dataframe, filter_type='consecutive', threshold=1):
    """
    Filters rows from the consensus_df that contain too many missing values,
    according to total or consecutive mode

    Parameters:
        dataframe: pandas dataframe
            dataframe containing descriptive and numerical columns to be
            filtered. Numerical columns must contain floats.
        filter_type: string
            may be either consecutive or total. Consecutive filters out rows
            which have two consecutive NaNs, while total filters out rows
            that do not have at least threshold values
        threshold: int
            if filter_type is total, number of NaNs allowed in numeric columns
            if filter_type is consecutive, number of NaNs allowed consecutively

    Returns:
        dataframe_filtered: DataFrame
            dataframe containing rows that meet the filtering condition
    """
    #filter_type = filter_type.lower

    if filter_type == 'total':
        logger.info('Total filtering active')
        #to adjust threshold number according to what is required by dropna
        #dropna is the number of NaNs allowed
        threshold = dataframe.shape[1] - threshold
        #to take only peptides with less that the threshold number of NaNs
        dataframe_filtered = dataframe.dropna(thresh=threshold)

    elif filter_type == 'consecutive':
        logger.info('Consecutive filtering active')
        #to filter out more than one consecutive NaN for each peptides
        dataframe_filtered = dataframe.copy()
        threshold += 1

        for index, row in dataframe.iterrows():
            for x in range (0, (len(row)-1)):
                if type(row[x]) is float:
                    if pd.isna(row[x:x+threshold]).values.all():
                        dataframe_filtered.drop(index, axis=0, inplace=True)
                        break

    else:
        dataframe_filtered = pd.DataFrame()
        logger.debug("Filter type was not recognised. Please try 'consecutive' or 'total'.")

    return dataframe_filtered


"""---------------------Statistical tests-------------------"""

def t_test_pair(df, cols_a, cols_b):
    """Completes paired t-test for each row of values, comparing those in col_a to those in col_b
    Parameters
    ----------
    df : dataframe
        Contains the complete dataset of interest
    cols_a : list
        list of column names to be included in the first set of values
    cols_b : list
        list of column names to be included in the second set of values
    Returns
    -------
    dataframe
        Returns entire input dataframe, with appended t-stat and p-value columns
    """

    for i in range(df.shape[0]):
        vals_a = df.loc[i, cols_a]
        vals_b = df.loc[i, cols_b]

        t_test_vals = stats.ttest_rel(vals_a, vals_b, axis=0)
        df.loc[i, 't-stat'] = t_test_vals[0]
        df.loc[i, 'p-value'] = t_test_vals[1]

    return df


def t_test_1samp(df, popmean, cols):
    """Completes one-sample t-test for each row of values, comparing those in cols to the popmean
    Parameters
    ----------
    df : dataframe
        Contains the complete dataset of interest
    cols : list
        list of column names to be included in the values
    popmean : float
        value to compare to (i.e. should be 0 or 1 for most ratio queries according to whether data has been logged (0) or not (1) prior to test)
    Returns
    -------
    dataframe
        Returns entire input dataframe, with appended t-stat and p-value columns
    """
    for i in range(df.shape[0]):
        vals = df.loc[i, cols]
        vals = vals.dropna()
        t_test_vals = stats.ttest_1samp(vals, popmean)
        df.loc[i, 't-stat'] = t_test_vals[0]
        df.loc[i, 'p-value'] = t_test_vals[1]
    return df


"""---------------------Fitting functions-------------------"""

def sigmoid(x, x0, k, a, c):
    """Sigmoid function, to be used for fitting parameters"""
    y = a / (1 + np.exp(-k*(x-x0))) + c
    return y

def linear(m, x, b):
    """linear function, to be used for fitting parameters"""
    y = m*x + b
    return y

def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def exponential(x, a, c, d):
    return a*np.exp(-c*x)+d

def fit_calculator(xdata, ydata, reg_function):
    """Calculates the best fit to a linear regression for the provided x&y data, using curvefit function."""
    popt, pcov = curve_fit(reg_function, xdata, ydata)
    #print (popt)

    x = np.linspace(min(xdata), max(xdata), 100)
    y = reg_function(x, *popt)
    return (x, y, xdata, ydata)

def fit_plotter(fit_dictionary, x_label, y_label):
    """Plots to x, y scatter plot, with the fitted curve overlayed."""
    fig = plt.figure()
    for key, value in fit_dictionary.items():
        x, y, x_data, y_data = value
        plt.scatter(x_data, y_data, label=key)
        plt.plot(x,y, label=key+'_fit')
        plt.xlim(min(x_data), max(x_data))
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
    plt.legend(loc='best')
    return fig
