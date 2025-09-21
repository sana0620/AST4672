#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 20:34:29 2025

@author: sana
"""

import numpy as np
import matplotlib.pyplot as plt
 
data_ccd = np.zeros(400)
# Draw 396 from poisson distributionOf 10000 lambda
data_ccd[:396]  = np.random.poisson(lam=10000, size=396)
print ('\nstd dev of poisson = ',np.std(data_ccd),'\n')

# Draw 4 values from a uniform distribution between 0 and 1e6
data_ccd[396:] = np.random.uniform(low=0, high=1e6, size=4) 
plt.figure(1) 
plt.plot(data_ccd)
plt.title('Given data set')
plt.grid()
plt.savefig('Given data set')

mean_ccd = np.mean(data_ccd)
median_ccd  = np.median(data_ccd)
std_ccd     = np.std(data_ccd)
print("mean =  " ,mean_ccd)
print("median =  ",median_ccd)
print("std dev =  ",std_ccd, '\n\n')


ccd_new_data = data_ccd[np.where(abs(data_ccd - median_ccd) < 5*std_ccd)]
plt.figure(2) 
plt.plot(ccd_new_data)
plt.title('Masking Once with 5sigma')
plt.grid()
plt.savefig('Masking once with 5 sigma')
print ( "mean new =    ", np.mean(ccd_new_data))
print ( "median new =    ", np.median(ccd_new_data))
print ( "std dev new =    ", np.std(ccd_new_data), '\n\n')

print('The new mean, median and std dev are printed above \n since the new data array has been arranged within 5 sigma from the median, The new median is same for the data \n The new mean ans std deviation are completely different and much smaller\n\n')



#the new sub subsample is

ccd_data2 = ccd_new_data[np.where(abs(ccd_new_data - np.median(ccd_new_data)) < 5*np.std(ccd_new_data))]
print ( "mean new 2=    ", np.mean(ccd_data2))
print ( "median new 2=    ", np.median(ccd_data2))
print ( "std dev new 2=    ", np.std(ccd_data2),'\n')
plt.figure(3) 
plt.plot(ccd_data2)
plt.title('Masking twice 5 sigma')
plt.grid()
plt.savefig('Masking twice with 5 sigma')

print('The median for the new data set is same \n The data points within 5 times the std dev of the subsample are clipped \n The new data set created has even less standard deviation\n\n')


def sigrej(data, sigma_limits, masking_times, mask=None):
    if mask is None:
        mask = [True] * len(data)
    mask = np.array(mask, dtype=bool)

    for _ in range(masking_times):
        good_data = data[mask]
        if len(good_data) == 0:
            print("All data points rejected. Returning empty mask.")
            return mask
        median = np.median(good_data)
        std = np.std(good_data)
        for i in range(len(data)):
            if mask[i] and abs(data[i] - median) > sigma_limits * std:
                mask[i] = False

    return mask


final_mask = sigrej(data_ccd, 5, 2)
cleaned_data = data_ccd[final_mask]

plt.figure()
plt.plot(cleaned_data)
plt.title('Cleaned Data using the function')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.savefig('Masked with function.png')

print ( "function reduced data mean =    ", np.mean(cleaned_data))
print ( "function reduced data median =    ", np.median(cleaned_data))
print ( "function reduced data std dev =    ", np.std(cleaned_data),'\n')











