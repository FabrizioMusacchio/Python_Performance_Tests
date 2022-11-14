"""
A script to evaluate the collected results from my IDE peformance test.

author: Fabrizio Musacchio (fabriziomusacchio.com)
date:   Nov 11, 2022
"""
# %% IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
import os
# %% PATHS
zarr_filename = "data.zarr"
# %% PARAMETERS
colors={"macOS VS Code (interactive)": "cornflowerblue", 
        "macOS VS Code (terminal)": "navy"}
linestyles = {"macOS VS Code (interactive)": "-", 
              "macOS VS Code (terminal)": "-"}
# %% FUNCTIONS
def plot_1D_series(times, y, y_err, figsize=(5,3.5), xlim=(0,1),ylim=(0,1),
                   xticks=np.arange(0,1),yticks=np.arange(0,1),title="",
                   test="", OS="", editor="", show_plot=False):
    fig=plt.figure(1, figsize=figsize)
    plt.close(1)
    fig.clf()
    plt.plot(times, y, label=editor+" ("+OS+")", lw=2, c="royalblue")
    plt.fill_between(times, y - y_err, y + y_err,
                    edgecolor="cornflowerblue", facecolor="cornflowerblue",
                    alpha=0.25, linewidth=0.0)
    plt.text(times[-1:], y[-1:], str(y[-1:].round(2)[0])+" s ⟶ ",
             ha="right", va="center", clip_on=False)
    plt.xlabel("N", fontsize=12, fontweight="bold")
    plt.ylabel("time [s]", fontsize=12, fontweight="bold")
    plt.title(title, fontweight="bold")
    plt.legend(loc="upper left")
    axis = plt.gca()
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['bottom'].set_linewidth(2)
    axis.spines['left'].set_linewidth(2)
    axis.xaxis.set_tick_params(width=2, length=8)
    axis.yaxis.set_tick_params(width=2, length=8)
    plt.xlim(xlim)
    plt.xticks(xticks,fontsize=12)
    plt.ylim(ylim)
    plt.yticks(yticks,fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/"+test+" "+OS+" "+editor+".pdf")
    if show_plot:
        plt.show()

def plot_multiple_1D_series(times, y, y_err, zarr_groups, colors, linestyles,
                            figsize=(5,3.5), xlim=(0,1),ylim=(0,1), 
                            xticks=np.arange(0,1),yticks=np.arange(0,1),title="",
                            test="", show_plot=False, round_precision=2):
    plt.close(1)
    fig=plt.figure(1, figsize=figsize)
    fig.clf()
    for group_i in range(times.shape[0]):
        plt.plot(times[group_i], y[group_i], label=zarr_groups[group_i], lw=2, alpha=0.75,
                 c=colors[zarr_groups[group_i]], ls=linestyles[zarr_groups[group_i]])
        plt.fill_between(times[group_i], y[group_i] - y_err[group_i], 
                         y[group_i] + y_err[group_i],
                        edgecolor=colors[zarr_groups[group_i]], 
                        facecolor=colors[zarr_groups[group_i]],
                        alpha=0.25, linewidth=0.0)
        plt.text(times[group_i, -1:], y[group_i,-1:], 
                 " ⟵ " + str(y[group_i, -1:].round(round_precision)[0])+" s",
                 color=colors[zarr_groups[group_i]],
                 ha="left", va="center", clip_on=False)
    plt.xlabel("N", fontsize=12, fontweight="bold")
    plt.ylabel("time [s]", fontsize=12, fontweight="bold")
    plt.title(title, fontweight="bold")
    plt.legend(loc="upper left")
    axis = plt.gca()
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['bottom'].set_linewidth(2)
    axis.spines['left'].set_linewidth(2)
    axis.xaxis.set_tick_params(width=2, length=8)
    axis.yaxis.set_tick_params(width=2, length=8)
    plt.xlim(xlim)
    plt.xticks(xticks,fontsize=12)
    plt.ylim(ylim)
    plt.yticks(yticks,fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/"+test+".pdf")
    if show_plot:
        plt.show()

# %% COLLECT STORES DATA
zarr_in = zarr.open(zarr_filename, mode="r")
print(zarr_in.info)
zarr_groups = []
zarr_df = pd.DataFrame()
for key in zarr_in.group_keys():
    zarr_groups.append(key)
    zarr_df_tmp = pd.DataFrame(index=[0])
    zarr_df_tmp["OS"]     = zarr_in[key].attrs["OS"]
    zarr_df_tmp["editor"] = zarr_in[key].attrs["editor"]
    zarr_df_tmp["N_rep"]  = zarr_in[key].attrs["N_rep"]
    zarr_df_tmp["N exponentiation"]  = zarr_in[key]["exponential_test"].attrs["N"]
    zarr_df_tmp["N transpose"]       = zarr_in[key]["transpose_test"].attrs["N"]
    zarr_df_tmp["N svd"]             = zarr_in[key]["svd_test"].attrs["N"]
    zarr_df = pd.concat([zarr_df, zarr_df_tmp], ignore_index=True)
N_rep = zarr_df.iloc[0]["N_rep"]
# %% MAIN

# Evaluate the exponentiation test:
N     = zarr_df.iloc[0]["N exponentiation"]
times_mean = np.zeros((len(zarr_groups),3,N), dtype="float64")
times_std  = np.zeros((len(zarr_groups),3,N), dtype="float64")
for group_i, group in enumerate(zarr_groups):
    """ 
    group_i = 0
    group = zarr_groups[group_i] 
    """
    times_mean[group_i] = np.mean(zarr_in[group+"/exponential_test"], axis=0)
    times_std[group_i] = np.std(zarr_in[group+"/exponential_test"], axis=0)

    """ 
    plot_1D_series(times=times_mean[group_i,0, :], y=times_mean[group_i,1, :],
                y_err=times_std[group_i,1, :], figsize=(6,3.5), 
                xlim=(0,N+50), xticks=np.arange(0,N+1,20000), 
                ylim=(0,0.000250), yticks=np.arange(0,0.000251,0.000020),
                title=f"Average time for calculating $e^N$ ({N_rep} reps)",
                test="exponentiation recover", OS=zarr_df.iloc[group_i]["OS"],
                editor=zarr_df.iloc[group_i]["editor"])
    """

plot_multiple_1D_series(times=times_mean[:,0, :], y=times_mean[:,1, :],
                y_err=times_std[:,1, :], figsize=(8,4.5),
                xlim=(0,N+50), xticks=np.arange(0,N+1,20000),
                ylim=(0,0.000350), yticks=np.arange(0,0.000351,0.000030),
                title=f"Average time for calculating $e^N$ ({N_rep} reps)",
                test="all exponentiation", zarr_groups=zarr_groups,
                colors=colors, linestyles=linestyles, round_precision=2)

# Evaluate the transposing test:
N     = zarr_df.iloc[0]["N transpose"]
times_mean = np.zeros((len(zarr_groups),3,N), dtype="float64")
times_std  = np.zeros((len(zarr_groups),3,N), dtype="float64")
for group_i, group in enumerate(zarr_groups):
    times_mean[group_i] = np.mean(zarr_in[group+"/transpose_test"], axis=0)
    times_std[group_i] = np.std(zarr_in[group+"/transpose_test"], axis=0)

plot_multiple_1D_series(times=times_mean[:,0, :], y=times_mean[:,1, :],
                y_err=times_std[:,1, :], figsize=(8,4.5),
                xlim=(0,2050), xticks=np.arange(0,2001,250), 
                ylim=(0,0.020), yticks=np.arange(0,0.021,0.002),
                title=f"Average time for allocating and\ntransposing an $N \\times N$ array ({N_rep} reps)",
                test="all transposing", zarr_groups=zarr_groups,
                colors=colors, linestyles=linestyles, round_precision=2)

# Evaluate the SVD test:
N     = zarr_df.iloc[0]["N svd"]
times_mean = np.zeros((len(zarr_groups),3,N), dtype="float64")
times_std  = np.zeros((len(zarr_groups),3,N), dtype="float64")
for group_i, group in enumerate(zarr_groups):
    times_mean[group_i] = np.mean(zarr_in[group+"/svd_test"], axis=0)
    times_std[group_i] = np.std(zarr_in[group+"/svd_test"], axis=0)

plot_multiple_1D_series(times=times_mean[:,0, :], y=times_mean[:,1, :],
                y_err=times_std[:,1, :], figsize=(8,4.5),
                xlim=(0,N+5), xticks=np.arange(0,N+1,50), 
                ylim=(0,0.35), yticks=np.arange(0,0.36,0.05),
                title=f"Average time for SVD of an $N \\times N$ array ({N_rep} reps)",
                test="all svd", zarr_groups=zarr_groups,
                colors=colors, linestyles=linestyles, round_precision=2)


# %% END


