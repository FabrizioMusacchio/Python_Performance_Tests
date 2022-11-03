"""
Script to test the performance of different Python IDE.

author: Fabrizio musacchio (fabriziomusacchio.com)
date: 2022, Nov 2

for testing, create a virtual environment with conda:

conda create -n python_performance_tests python=3.9
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
# import matplotlib
# matplotlib.use('TkAgg')
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
import timeit
Process_t1 = calc_process_time(Process_t0, leadspaces="imports: ", output=True, unit="sec", verbose=True)
# %% NUMPY EXPONENTIAL FUNCTION
N_rep = 10
N=100000
times_exp = np.zeros((N_rep, 3,N), dtype="float64")
for rep in range(N_rep):
    times_exp_tmp = np.zeros((3,N), dtype="float64")
    for i in range(N):
        Process_t0 = time.time()
        dump = np.exp(i)
        del dump
        times_exp_tmp[0, i] = i
        times_exp_tmp[1,i] = calc_process_time(Process_t0, output=True, unit="sec")
        if i > 0:
            times_exp_tmp[2, i] = times_exp_tmp[2, i - 1] + times_exp_tmp[1, i]
    times_exp[rep,:,:] = times_exp_tmp
times_exp = times_exp.mean(axis=0)
#print(f"total elapsed time {N}-times calculating the exponential function: {times_exp[1,:].sum()}")
print(f"total elapsed time {N}-times calculating the exponential function: {times_exp[2,N-1]}")

fig=plt.figure(1)
fig.clf()
plt.plot(times_exp[0, :], times_exp[1, :], label='elapsed time')
plt.xlabel("N")
plt.ylabel("time [s]")
plt.title("Time for allocating and transposing a NxN array with NumPy",
          fontweight="bold")
plt.show()

fig=plt.figure(2)
fig.clf()
plt.plot(times_exp[0, :], times_exp[2, :], label='elapsed time')
plt.xlabel("N")
plt.ylabel("time [s]")
plt.title("Time for allocating and transposing a NxN array with NumPy",
          fontweight="bold")
plt.show()
# %% NUMPY ALLOCATE ARRAY MEMORY
N_rep=10
N=2000
times_allocate = np.zeros((N_rep,3,N), dtype="float64")
for rep in range(N_rep):
    times_allocate_tmp = np.zeros((3,N), dtype="float64")
    np.random.seed(1)
    for i in range(N):
        Process_t0 = time.time()
        dump = np.random.random((i,i))
        dump = np.transpose(dump)
        del dump
        times_allocate_tmp[0, i] = i
        times_allocate_tmp[1,i] = calc_process_time(Process_t0, output=True, unit="sec")
        if i>0:
            times_allocate_tmp[2, i] = times_allocate_tmp[2, i-1]+ times_allocate_tmp[1,i]
    times_allocate[rep,:, :] = times_allocate_tmp
times_allocate = times_allocate.mean(axis=0)
print(f"total elapsed for N={N}: {times_allocate[2,N-1]}")
#print(f"total elapsed for N={N}: {times_allocate[1,:].sum()}")
fig=plt.figure(1)
fig.clf()
plt.plot(times_allocate[0, :], times_allocate[1, :], label='elapsed time')
plt.xlabel("N")
plt.ylabel("time [s]")
plt.title("Time for allocating and transposing a NxN array with NumPy",
          fontweight="bold")
plt.show()

fig=plt.figure(2)
fig.clf()
plt.plot(times_allocate[0, :], times_allocate[2, :], label='elapsed time')
plt.xlabel("N")
plt.ylabel("time [s]")
plt.title("Time for allocating and transposing a NxN array with NumPy",
          fontweight="bold")
plt.show()
# %% END