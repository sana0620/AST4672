#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 22:09:14 2025

@author: sana
"""

#import the libraries required for proper execution
#import scipy as sp
#import matplotlib as mpl
#import matplotlib.pyplot as plt  
#import astropy.io.fits as fits
import numpy as np
from hw3_supportfile_sana import square, squareplot

#TEST FOR SQUARE
# Test 1: Integer array
test_square_1 = np.arange(10)
print("test_square_1 squared:", square(test_square_1))

# Test 2: 5x5 float array
test_square_2 = np.linspace(0, 25, 25).reshape(5, 5)
print("test_square_2 squared:", square(test_square_2))

#TEST FOR SQUAREPLOT
squareplot(1, 7, 25, saveplot="square_plot_function_sana.pdf")

