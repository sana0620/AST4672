#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 10:19:32 2025

@author: sana

support function file

"""
 
#import the libraries required for proper execution

#import scipy as sp
#import matplotlib as mpl 
#import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt 


#creating first function

def square(x):
    """
    This function returns the square of the input, which can be a scalar or a numpy array
    of any dimension and numerical type.

    Parameters
    ----------
    x : int, float, or numpy.ndarray
        A scalar or array to be squared.

        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.  We can also refer to
        variables like `var1`.
    Returns 
    -------
    int, float, or numpy.ndarray
        The square of the input. 
    Raises
    ------
    ValueError
        If the input is not a number or an array of numbers.


    Examples
    >>> square(3)
    9
    >>> square(np.array([1, 2, 3]))
    array([1, 4, 9])
    >>> square(np.array([[1.0, 2.0], [3.0, 4.0]]))
    array([[ 1.,  4.],
           [ 9., 16.]])
    """
    arr = np.array(x)
    return arr*arr


#creating Second function

def squareplot(low, high, num_points, saveplot=False):
    '''
    This function returns plot of squared numbers over a range as a pdf

    Parameters
    ----------
    low : float
        The lower bound of the range.
    high : float
        The upper bound of the range (inclusive).
    num_points : int
        Number of points to plot.
    saveplot : str 
        If a string is provided, saves the plot as a PDF with that filename.

    Returns
    -------
    plot

   
    Example
    -------
    >>> squareplot(1, 7, 5)
    >>> squareplot(1, 7, 5, saveplot="squareplot_hw3_sana.pdf")
    '''
    x = np.linspace(low, high, num_points)
    y = square(x)

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Input range")
    plt.ylabel("square")
    plt.title("Square Function")
    plt.grid(True)

    if saveplot:
        plt.savefig(saveplot, format='pdf')

    plt.show()












