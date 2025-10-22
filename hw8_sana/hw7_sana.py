#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 14:39:43 2025

@author: sana
"""
import os
import numpy as np
from astropy.io import fits

# Taking the data from the folder
folder_path = "hw7_data_export"
fits_files = [f for f in os.listdir(folder_path) if f.endswith(".fits")]

dark_files = sorted([f for f in fits_files if f.startswith("dark_13s_")])
object_files = sorted([f for f in fits_files if f.startswith("stars_13s_")])

dark_stack = []
for file in dark_files:
    with fits.open(os.path.join(folder_path, file)) as hdul:
        dark_stack.append(hdul[0].data)
dark_stack = np.array(dark_stack)

# Median combine the dark frames
median_dark = np.median(dark_stack, axis=0)

print("\nMedian dark pixel [217, 184]:", median_dark[217, 184],'\n')

# Get header from the first dark frame and add HISTORY
with fits.open(os.path.join(folder_path, dark_files[0])) as hdul:
    header = hdul[0].header
header.add_history("A Median combination dark frame is created.")


# Save the median dark frame to a new FITS file
fits.writeto("dark_13s_med.fits", median_dark, header=header, overwrite=True)


# Subtract the median dark from each object frame
object_subtracted = []
for file in object_files:
    with fits.open(os.path.join(folder_path, file)) as hdul:
        obj_data = hdul[0].data
        subtracted = obj_data - median_dark
        object_subtracted.append(subtracted)

# Print pixel value [217, 184] before and after subtraction for the first object frame
with fits.open(os.path.join(folder_path, object_files[0])) as hdul:
    original_pixel = hdul[0].data[217, 184]
print("\nOriginal stars_13s_1.fits pixel [217, 184]:", original_pixel,'\n')
print("\nSubtracted stars_13s_1.fits pixel [217, 184]:", object_subtracted[0][217, 184],'\n')

# Save the first subtracted object frame to a new FITS file
fits.writeto("hw7_prob2_graph1_sana.fits", object_subtracted[0], overwrite=True)