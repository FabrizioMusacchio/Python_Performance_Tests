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
try:
    os.mkdir("plots/evaluation/")
except:
    pass
try:
    os.mkdir("plots/evaluation/png")
except:
    pass
# %% PARAMETERS
colors={"VS Code (interactive)": "deepskyblue", 
        "VS Code (terminal)": "mediumblue",
        "PyCharm": "darkseagreen", 
        "Jupyter": "darkorange"}
linestyles = {"VS Code (interactive)": "-", 
              "VS Code (terminal)": "-",
              "PyCharm": "-", 
              "Jupyter": "-"}
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

def plot_multiple_1D_series(df_in, zarr_in, colors, linestyles, cumulative=False,
                            figsize=(5,3.5), xlim_add=5, xticks_steps=20000,
                            ylim=(0,1), yticks=np.arange(0,1),title="",
                            test="exponential_test", show_plot=False, round_precision=2):
    if cumulative:
        plot_axis = 2
    else:
        plot_axis = 1
    for OS in df_in["OS"].unique():
        #OS = df_in["OS"].unique()[0]
        curr_os_sub_df = df_in[df_in["OS"]==OS]
        for venv in curr_os_sub_df["venv"].unique():
            #venv = curr_os_sub_df["venv"].unique()[0]
            plt.close(1)
            fig=plt.figure(1, figsize=figsize)
            fig.clf()
            curr_os_venv_sub_df = curr_os_sub_df[curr_os_sub_df["venv"]==venv]
            for editor in curr_os_venv_sub_df["editor"]:
                #editor=curr_os_venv_sub_df["editor"][0]
                curr_array = zarr_in[OS+" "+editor+" "+venv+"/"+test]
                plt.plot(curr_array[0,0,:], np.mean(curr_array[:,plot_axis,:], axis=0),
                         label=editor, lw=1.5, alpha=0.75,
                         c=colors[editor], ls=linestyles[editor])
                plt.fill_between(curr_array[0,0,:], 
                                 np.mean(curr_array[:,plot_axis,:], axis=0)-np.std(curr_array[:,plot_axis,:], axis=0),
                                 np.mean(curr_array[:,plot_axis,:], axis=0)+np.std(curr_array[:,plot_axis,:], axis=0),
                                edgecolor=colors[editor],facecolor=colors[editor],
                                alpha=0.25, linewidth=0.0)
                plt.text(curr_array[0,0,-1:], np.mean(curr_array[:,plot_axis,-1:], axis=0),
                         " ⟵ " + str(np.mean(curr_array[:,plot_axis,-1:], axis=0).round(round_precision)[0]) +" s",
                         color=colors[editor],
                         ha="left", va="center", clip_on=False)
            N=curr_array[0,0,:].shape[0]
            plt.xlabel("N", fontsize=12, fontweight="bold")
            plt.ylabel("time [s]", fontsize=12, fontweight="bold")
            plt.title(title+" ("+venv+")", fontweight="bold")
            plt.legend(loc="upper left")
            axis = plt.gca()
            axis.spines['right'].set_visible(False)
            axis.spines['top'].set_visible(False)
            axis.spines['bottom'].set_linewidth(2)
            axis.spines['left'].set_linewidth(2)
            axis.xaxis.set_tick_params(width=2, length=8)
            axis.yaxis.set_tick_params(width=2, length=8)
            plt.xlim(0,N+xlim_add)
            plt.xticks(np.arange(0,N+1,xticks_steps),fontsize=12)
            plt.ylim(ylim)
            plt.yticks(yticks,fontsize=12)
            plt.tight_layout()
            if cumulative:
                plotname = OS+" "+venv+" "+test+" (cumulative)"
            else:
                plotname = OS+" "+venv+" "+test
            plt.savefig("plots/evaluation/"+plotname+".pdf")
            plt.savefig("plots/evaluation/png/"+plotname+".png", dpi=200)
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
    zarr_df_tmp["venv"] = zarr_in[key].attrs["venv"]
    zarr_df_tmp["N_rep"]  = zarr_in[key].attrs["N_rep"]
    zarr_df_tmp["N exponentiation"]  = zarr_in[key]["exponential_test"].attrs["N"]
    zarr_df_tmp["N transpose"]       = zarr_in[key]["transpose_test"].attrs["N"]
    zarr_df_tmp["N svd"]             = zarr_in[key]["svd_test"].attrs["N"]
    zarr_df = pd.concat([zarr_df, zarr_df_tmp], ignore_index=True)
N_rep = zarr_df.iloc[0]["N_rep"]
# %% EVALUATE THE EXPONENTIATION TEST
plot_multiple_1D_series(df_in=zarr_df, zarr_in=zarr_in, figsize=(6,4.5),
                        cumulative=True, 
                        xlim_add=5, xticks_steps=20000,
                        ylim=(0,0.8), yticks=np.arange(0,0.81,0.1),
                        title=f"Average cumulative time for calculating $e^N$ ({N_rep} reps)",
                        test="exponential_test", 
                        colors=colors, linestyles=linestyles, round_precision=2)
plot_multiple_1D_series(df_in=zarr_df, zarr_in=zarr_in, figsize=(6,4.5),
                        cumulative=False, 
                        xlim_add=5, xticks_steps=20000,
                        ylim=(0,0.000350), yticks=np.arange(0,0.000351,0.000030),
                        title=f"Average time for calculating $e^N$ ({N_rep} reps)",
                        test="exponential_test", 
                        colors=colors, linestyles=linestyles, round_precision=2)
# %% EVALUATE THE TRANSPOSING TEST
plot_multiple_1D_series(df_in=zarr_df, zarr_in=zarr_in, figsize=(6,4.5),
                        cumulative=False, 
                        xlim_add=5, xticks_steps=250,
                        ylim=(0,0.050), yticks=np.arange(0,0.051,0.005),
                        title=f"Average time for allocating and\ntransposing an $N \\times N$ array ({N_rep} reps)",
                        test="transpose_test", 
                        colors=colors, linestyles=linestyles, round_precision=2)
# %% EVALUATE THE SVD TEST
plot_multiple_1D_series(df_in=zarr_df, zarr_in=zarr_in, figsize=(6,4.5),
                        cumulative=False, 
                        xlim_add=5, xticks_steps=50,
                        ylim=(0,0.35), yticks=np.arange(0,0.36,0.05),
                        title=f"Average time for SVD of an $N \\times N$ array ({N_rep} reps)",
                        test="svd_test", 
                        colors=colors, linestyles=linestyles, round_precision=2)
# %% END


