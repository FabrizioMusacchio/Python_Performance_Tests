"""
Script to test the performance of different Python IDE.

author: Fabrizio musacchio (fabriziomusacchio.com)
date: 2022, Nov 2

for testing, create a virtual environment with conda:

conda install -y matplotlib numpy seaborn pandas zarr numcodecs plotly scikit-image scipy
"""
# %% IMPORTS (ESSENTIAL)
import time
# %% FUNCTIONS  (ESSENTIAL)
def calc_process_time(t0, verbose=False, leadspaces="", output=False, unit="min"):
    """
        Calculates the processing time/time difference for a given input time and the current time

    Usage:
        Process_t0 = time.time()
        #your process
        calc_process_time(Process_t0, verbose=True, leadspaces="  ")

        :param t0:              starting time stamp
        :param verbose:         verbose? True or False
        :param leadspaces:      pre-fix, e.g., some space in front of any verbose output
        :param output:          provide an output (s. below)? True or False
        :return: dt (optional)  the calculated processing time
    :rtype:
    """
    dt = time.time() - t0
    if verbose:
        if unit=="min":
            print(leadspaces + f'process time: {round(dt / 60, 2)} min')
        elif unit=="sec":
            print(leadspaces + f'process time: {round(dt , 10)} sec')
    if output:
        return dt
# %% IMPORT TEST
Process_t0 = time.time()
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import zarr
from numcodecs import Blosc
import plotly
import plotly.express as px
import plotly.io as pio
from skimage import segmentation as seg
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from scipy import ndimage
Process_t1 = calc_process_time(Process_t0, leadspaces="imports: ", output=True, unit="sec", verbose=True)
# %% FUNCTIONS


# %% NUMPY EXPONENTIAL FUNCTION
N=100000
times_exp = np.zeros((2,N), dtype="float64")
for i in range(N):
    Process_t0 = time.time()
    dump = np.exp(i)
    times_exp[0, i] = i
    times_exp[1,i] = calc_process_time(Process_t0, output=True, unit="sec")
print(f"total elapsed time {N}-times calculating the exponential function: {times_exp[1,N-1]}")
# %% NUMPY ALLOCATE ARRAY MEMORY
N=10000
times_allocate = np.zeros((2,N), dtype="float64")
np.random.seed(1)
for i in range(N):
    Process_t0 = time.time()
    dump = np.random.random((i,i))
    dump = np.transpose(dump)
    del dump
    times_allocate[0, i] = i
    times_allocate[1,i] = calc_process_time(Process_t0, output=True, unit="sec")
print(f"total elapsed time {N}-times calculating the exponential function: {times_allocate[1,N-1]}")
fig=plt.figure(1)
fig.clf()
plt.plot(times_allocate[0, :], times_allocate[1, :], label='Sinus inLine')
# %%
