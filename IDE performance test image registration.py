"""
Script to test the performance of different Python IDE. This script contains one
specific test: image registration.

author: Fabrizio Musacchio (fabriziomusacchio.com)
date:   Nov 14, 2022
"""
# %% IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.transform import SimilarityTransform, warp
import os
import time
import zarr
# %% PARAMETERS
OS          = "Linux" # macOS Windows Linux
editor_n    = 2 # choose the editor# from the list below
venv_n      = 0 # choose venv form the list below
N_rep       = 10 # Number of repetitions
#                          0                     1               2         3
editor_list = ["VS Code (interactive)","VS Code (terminal)","PyCharm","Jupyter"]
#                  0       1          2
venv_list   = ["conda","python","virtualenv"]
editor      = editor_list[editor_n]
venv        = venv_list[venv_n]
# %% OPEN ZARR
zarr_filename = "data.zarr"
zarr_out_root   = zarr.open(zarr_filename, mode='a')
zarr_curr_group = zarr_out_root[OS + " " + editor + " " + venv]
# %% FUNCTIONS
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
# %% CREATE SOME TOY DATA
layers=100
image = data.eagle()
image_stack = np.zeros((layers, image.shape[0], image.shape[1]))
image_stack[0] = image
shifts_true = np.zeros((layers, 2))
np.random.seed(1)
for layer in range(1,layers):
    shift = (np.random.randint(-100, 100), np.random.randint(-100, 100))
    tform = SimilarityTransform(translation=shift)
    image_stack[layer] = warp(image, tform)
    shifts_true[layer] = shift
# %% MAIN
times_regs = []
for rep in range(N_rep):
    Process_t0 = time.time()
    print(f"iteration: {rep}", end="")
    image_stack_reg = np.zeros((layers, image.shape[0], image.shape[1]))
    shifts_detected = np.zeros((layers, 2))
    pearson_R       = np.zeros(layers)
    pearson_R_reg   = np.zeros(layers)
    for layer in range(0,layers):
        shift, _, _ = phase_cross_correlation(image, image_stack[layer])
        shifts_detected[layer] = (-shift[1], -shift[0])
        tform = SimilarityTransform(translation=shifts_detected[layer])
        image_stack_reg[layer] = warp(image_stack[layer], tform)
        pearson_R[layer]     = sp.stats.pearsonr(image[100:-100,100:-100].flatten(), 
                                                 image_stack[layer, 100:-100,100:-100].flatten())[0]
        pearson_R_reg[layer] = sp.stats.pearsonr(image[100:-100,100:-100].flatten(), 
                                                 image_stack_reg[layer, 100:-100,100:-100].flatten())[0]
    Process_t1 = calc_process_time(Process_t0, leadspaces=f"{N_rep} reps: ", 
                                    output=True, unit="sec", verbose=False)
    print(f" ({Process_t1} s)")
    times_regs.append(Process_t1)                         
print(f"average processing time: {np.mean(times_regs)} ± {np.std(times_regs)} s")
# %% WRITE THE RESULTS INTO THE ZARR COLLECTOR FILE
zarr_out_reg = zarr_curr_group.create_dataset("registration_test", data=times_regs,
                                              chunks=False, overwrite=True)
# %% CTRL PLOTS
""" 
plt.imshow(image_stack_reg[1])
plt.imshow(image_stack[1])
plt.imshow(image)
plt.imshow(image*(image_stack_reg[1]), cmap='gray')
plt.imshow(image-image_stack_reg[10]*100, cmap='gray')
plt.imshow((image_stack_reg[1]), cmap='gray')

plt.plot(shifts_true[:,0], shifts_true[:,1], 'd', c="k", label="true shifts")
plt.plot(-shifts_detected[:,0], -shifts_detected[:,1], '.', c="pink", label="detected shifts")
plt.legend()

plt.plot(pearson_R, label="correlation before registration", c="r")
plt.plot(pearson_R_reg, label="correlation after registration", c="g")
"""

# %% END
