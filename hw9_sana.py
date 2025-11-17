#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 14:39:43 2025

@author: sana
"""
import os
import numpy as np
from astropy.io import fits
import gaussian

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

print("\nhw7 results:\n\nMedian dark pixel [217, 184]:", median_dark[217, 184])

# Get header from the first dark frame and add HISTORY
with fits.open(os.path.join(folder_path, dark_files[0])) as hdul:
    header = hdul[0].header
header.add_history("A Median combination dark frame is created.")

fits.writeto("dark_13s_med.fits", median_dark, header=header, overwrite=True)

# Subtract the median dark from each object frame
object_dark_subtracted = []
for file in object_files:
    with fits.open(os.path.join(folder_path, file)) as hdul:
        obj_data = hdul[0].data
        subtracted = obj_data - median_dark
        object_dark_subtracted.append(subtracted)

with fits.open(os.path.join(folder_path, object_files[0])) as hdul:
    original_pixel = hdul[0].data[217, 184]
print("Original stars_13s_1.fits pixel [217, 184]:", original_pixel)
print("Subtracted stars_13s_1.fits pixel [217, 184]:", object_dark_subtracted[0][217, 184])

fits.writeto("hw7_prob2_graph1_sana.fits", object_dark_subtracted[0], overwrite=True)
obj_frames,ny,nx = np.shape(object_dark_subtracted)


''''----------------------------------------------------------------------'''


# HW8 ahead

print("\nhw8 results:\n")

def skycormednorm(object_frame, norm_sky_frame, norm_region=None):
    """ 
    Implements sky subtraction using a normalized sky frame.

    Parameters
    ----------
    object_frame : ndarray, 2D
        Dark-subtracted object frame.
    norm_sky_frame : ndarray, 2D
        Normalized sky frame (already divided by its median
        when it was created).
    norm_region : tuple, optional
        ((y1, x1), (y2, x2)) region used to compute the object
        median for normalization. Default: whole image.

    Returns
    -------
    sky_sub_obj : ndarray, 2D
        Sky-subtracted object frame.
    norm_factor : float
        Normalization factor applied to the normalized sky frame.
    """

    obj = np.copy(object_frame)
    sky = np.copy(norm_sky_frame)
    ny, nx = obj.shape
    if norm_region is None:
        norm_region = ((0, 0), (ny, nx))
    (y1, x1), (y2, x2) = norm_region

    norm_factor = np.median(obj[y1:y2, x1:x2])
    denorm_sky = sky * norm_factor
    sky_sub_obj = obj - denorm_sky
    
    return sky_sub_obj, norm_factor

# Open the sky frame created in practicum5
with fits.open("sky_13s_mednorm.fits") as hdul:
        sky_frame = hdul[0].data

# Initialize arrays
norm_factors_list = []
sky_sub_frames = []
norm_region = ((225, 225), (-225, -225))

# Apply skycormednorm to object frames in a loop
for i in range(obj_frames):
    sky_sub_obj,norm_factor = skycormednorm(object_dark_subtracted[i],sky_frame,norm_region)
    norm_factors_list.append(norm_factor)
    sky_sub_frames.append(sky_sub_obj)

norm_factors = np.array(norm_factors_list) # Store factors

last_frame = sky_sub_frames[-1]
fits.writeto('stars_13s_9_nosky.fits', last_frame, overwrite=True)

print("Last frame is saved as stars_13s_9_nosky.fits")
print(f"Last frame pixel [217, 184] value: {last_frame[217, 184]}\n\n")

# the newwest array is dark subtracted and sky corrected
# named sky_sub_frames



''''----------------------------------------------------------------------'''

#  hw9 ahead
print("HW9 ahead \n")

#======================== HW9: Photometry ================

# Question 2: 
# Parameters are yet to be calculated
# Parameter array is defined with NaNs below
# axes: [star, frame, parameter]
# parameters: [yguess, xguess, width, cy, cx, star, sky]

photometry = np.array([
    [ [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # star 0, frame 0
      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # star 0, frame 1
      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] ],# star 0, frame 2

    [ [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # star 1, frame 0
      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # star 1, frame 1
      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] ],# star 1, frame 2

    [ [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # star 2, frame 0
      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # star 2, frame 1
      [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] ] # star 2, frame 2
], dtype=float)

# Given positions in frame 0 (stars_13s_1) from the example table in hw9 PDF:
photometry[0, 0, 0] = 698.0 #star1
photometry[0, 0, 1] = 512.0
photometry[1, 0, 0] = 668.0 #star2
photometry[1, 0, 1] = 520.0
photometry[2, 0, 0] = 568.0 #star3
photometry[2, 0, 1] = 283.0

# Known star 0 position in frame 0 (from the HW9 instructions)
y0_f0 = photometry[0, 0, 0]   # 698
x0_f0 = photometry[0, 0, 1]   # 512

# Star finder function as used in midterm
def find_star_peak(frame, y_guess, x_guess, search_halfsize=20):
    """
    Find the brightest pixel near an approximate position (y_guess, x_guess)
    within a square region of +/- search_halfsize pixels.
    """
    ny, nx = frame.shape
    y0 = int(round(y_guess))
    x0 = int(round(x_guess))

    y1 = max(0, y0 - search_halfsize)
    y2 = min(ny, y0 + search_halfsize + 1)
    x1 = max(0, x0 - search_halfsize)
    x2 = min(nx, x0 + search_halfsize + 1)

    sub = frame[y1:y2, x1:x2]
    # Index of brightest pixel in the subimage
    dy, dx = np.unravel_index(np.argmax(sub), sub.shape)

    # Convert back to full-frame coordinates
    y_peak = y1 + dy
    x_peak = x1 + dx
    return float(y_peak), float(x_peak)

# Use the first three corrected frames:
y0_f1, x0_f1 = find_star_peak(sky_sub_frames[1], y0_f0, x0_f0)
y0_f2, x0_f2 = find_star_peak(sky_sub_frames[2], y0_f0, x0_f0)

print("Star 0 in frame 0 (given):", y0_f0, x0_f0)
print("Star 0 in frame 1 (found):", y0_f1, x0_f1)
print("Star 0 in frame 2 (found):", y0_f2, x0_f2)

# Put these into the table:
photometry[0, 1, 0] = y0_f1
photometry[0, 1, 1] = x0_f1
photometry[0, 2, 0] = y0_f2
photometry[0, 2, 1] = x0_f2

# Computing offsets of star 0 between frames 1 and 2
dy_01 = photometry[0, 1, 0] - photometry[0, 0, 0]  # frame1 - frame0
dx_01 = photometry[0, 1, 1] - photometry[0, 0, 1]
dy_02 = photometry[0, 2, 0] - photometry[0, 0, 0]  # frame2 - frame0
dx_02 = photometry[0, 2, 1] - photometry[0, 0, 1]

# Apply same offsets to stars 1 and 2 to get the coordinates
for star in [1, 2]:
    # added in frame 1
    photometry[star, 1, 0] = photometry[star, 0, 0] + dy_01
    photometry[star, 1, 1] = photometry[star, 0, 1] + dx_01
    # added in frame 2
    photometry[star, 2, 0] = photometry[star, 0, 0] + dy_02
    photometry[star, 2, 1] = photometry[star, 0, 1] + dx_02

# Question 3: 
# a.Gaussian fit to get fitting parameters
box_halfsize = 15   # half-size of fitting subimage (~31x31 pixels)
nstar = photometry.shape[0]
nframe = 3  # using first three frames only
ny, nx = object_dark_subtracted[0].shape

for s in range(nstar):
    for f in range(nframe):
        yguess = photometry[s, f, 0]
        xguess = photometry[s, f, 1]

        if np.isnan(yguess) or np.isnan(xguess):
            continue  # skip if not filled
            
        y0 = int(round(yguess))
        x0 = int(round(xguess))

        frame = object_dark_subtracted[f]

        # Define subimage bounds (clipped to image)
        y1 = max(0, y0 - box_halfsize)
        y2 = min(ny, y0 + box_halfsize + 1)
        x1 = max(0, x0 - box_halfsize)
        x2 = min(nx, x0 + box_halfsize + 1)

        sub = frame[y1:y2, x1:x2].copy()

        # Subtract median to get ~zero background
        sub -= np.median(sub)

        # To prevent the gaussian function from stopping when it hits a diagonal values fitting error
        try:
            # Fit 2D Gaussian
            width, center, height, err = gaussian.fitgaussian(sub)
        except: 
            # leave NaNs in photometry for this star/frame
            continue

        sigy, sigx = width
        cy_sub, cx_sub = center
        cy = cy_sub + y1
        cx = cx_sub + x1

        avg_width = 0.5 * (sigy + sigx)

        photometry[s, f, 2] = avg_width   # width
        photometry[s, f, 3] = cy          # fitted y
        photometry[s, f, 4] = cx          # fitted x

# Add all gaussian fits data to the array
valid_widths = photometry[:, :, 2][~np.isnan(photometry[:, :, 2])]
print("\nFitted widths (per star/frame):")
print(photometry[:, :, 2])
print('\n\nThe updated array is:\n',photometry[:, :, :]) # to be copied in log file

# b. Aperture Phtometry parameters

all_widths = photometry[:, :, 2]
# Keep only finite widths and discard crazy large ones, i.e more than 5 pixels
good = np.isfinite(all_widths) & (all_widths > 0) & (all_widths < 5)
good_widths = all_widths[good]
mean_width = np.mean(good_widths)
print("\nGood widths used for average:", good_widths)
print(f'\nRobust average width: {mean_width:.4f}') 

r_ap = 1.5
r_in  = 3.0 * mean_width   
r_out = 5.0 * mean_width   
subsize = 8 * mean_width
