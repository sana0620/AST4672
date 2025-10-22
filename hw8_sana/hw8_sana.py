#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 14:39:43 2025

@author: sana
"""
import os
import numpy as np
from astropy.io import fits



''''------------------------------------------------------------------'''

# HW7 steps to obtain dark subtracted object frame

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

print("\nhw7 results\n\nMedian dark pixel [217, 184]:", median_dark[217, 184])

# Get header from the first dark frame and add HISTORY
with fits.open(os.path.join(folder_path, dark_files[0])) as hdul:
    header = hdul[0].header
header.add_history("A Median combination dark frame is created.")


# Save the median dark frame to a new FITS file
fits.writeto("dark_13s_med.fits", median_dark, header=header, overwrite=True)


# Subtract the median dark from each object frame
object_dark_subtracted = []
for file in object_files:
    with fits.open(os.path.join(folder_path, file)) as hdul:
        obj_data = hdul[0].data
        subtracted = obj_data - median_dark
        object_dark_subtracted.append(subtracted)

# Print pixel value [217, 184] before and after subtraction for the first object frame
with fits.open(os.path.join(folder_path, object_files[0])) as hdul:
    original_pixel = hdul[0].data[217, 184]
print("Original stars_13s_1.fits pixel [217, 184]:", original_pixel)
print("Subtracted stars_13s_1.fits pixel [217, 184]:", object_dark_subtracted[0][217, 184])

# Save the first subtracted object frame to a new FITS file
fits.writeto("hw7_prob2_graph1_sana.fits", object_dark_subtracted[0], overwrite=True)

obj_frames,ny,nx = np.shape(object_dark_subtracted)


''''----------------------------------------------------------------------'''


# HW8 ahead

print("\nhw8 results\n")

def skycormednorm(object_frame, norm_sky_frame, norm_region=None):
    """ 
    Implements sky subtraction using normalized sky frame.
    
    Parameters:
    -----------
    object_frame : ndarray
        2D array of dark-subtracted object frame.
    sky_frame : ndarray
        2D array of dark-subtracted normalized sky frame.    
    norm_region : tuple, optional
        ((y1, x1), (y2, x2)) defining normalization region.
        Default: entire image.
    
    Returns:
    --------
    sky_normalized_object : ndarray
        2D normalized, sky subtracted object image.
    norm_factors : ndarray
        1D array of normalization factors for each frame.
    """
    obj_copy = np.copy(object_frame)
    sky_copy = np.copy(norm_sky_frame)
    
    ny, nx = obj_copy.shape
    
    # Default normalization region is full image
    if norm_region is None:
        norm_region = ((0, 0), (ny, nx))
    
    (y1, x1), (y2, x2) = norm_region
    
    
    # Compute normalization factors
    
    obj_region = obj_copy[y1:y2, x1:x2]
    sky_region = sky_copy[y1:y2, x1:x2]
    norm_factor  = np.median(obj_region)/np.median(sky_region)
    
    # Denormalize the sky frame
    
    denormalized_sky = sky_copy * norm_factor
    
    # Compute the sky-subtracted object frame
    
    sky_sub_obj = obj_copy - denormalized_sky
    
    return sky_sub_obj, norm_factor

# Open the sky frame created in practicum5

with fits.open("sky_13s_mednorm.fits") as hdul:
        sky_frame = hdul[0].data

# Initialize arrays
norm_factors_list = []
sky_sub_frames = []

# Apply skycormednorm to object frames in a loop

for i in range(obj_frames):
    sky_sub_obj,norm_factor = skycormednorm(object_dark_subtracted[i],sky_frame)
    norm_factors_list.append(norm_factor)
    sky_sub_frames.append(sky_sub_obj)

norm_factors = np.array(norm_factors_list) # Store factors

last_frame = sky_sub_frames[-1]

# Save the last frame to a FITS file
fits.writeto('stars_13s_9_nosky.fits', last_frame, overwrite=True)

print("Last frame is saved as stars_13s_9_nosky.fits")

# Print the value of pixel [217, 184]
print(f"Last frame pixel [217, 184] value: {last_frame[217, 184]}")

