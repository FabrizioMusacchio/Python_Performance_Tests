"""
A script to evaluate the collected results from my IDE peformance test.

author: Fabrizio Musacchio (fabriziomusacchio.com)
date:   Nov 11, 2022
"""
# %% IMPORTS
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
import os
import pingouin as pg
# %% PATHS
zarr_filename = "data.zarr"
try:
    os.mkdir("plots/evaluation/")
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
plotlabels = {"VS Code (interactive)": "VS Code$_i$", 
              "VS Code (terminal)": "VS Code$_t$",
              "PyCharm": "PyCharm", 
              "Jupyter": "Jupyter",
              "conda": "conda",
              "python": "python",
              "virtualenv": "virtualenv"}
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
            plt.title(title+" "+venv, fontweight="bold")
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
            if show_plot:
                plt.show()

def boxplot_editors_per_os_per_venv(df_in, zarr_in, colors, plotlabels,
                                    cumulative=False, figsize=(5,3.5), lw_boxplots = 2.5,
                                    ylim=(0,1), yticks=np.arange(0,1),title="",ylabel="",
                                    test="exponential_test", show_plot=False,
                                    statsOffset=1, barOffset=0.1, boxwidth=1):
    if cumulative:
        plot_axis = 2
    else:
        plot_axis = 1
    for OS in df_in["OS"].unique():
        #OS = df_in["OS"].unique()[0]
        curr_os_sub_df = df_in[df_in["OS"]==OS]
        for venv in curr_os_sub_df["venv"].unique():
            #venv = curr_os_sub_df["venv"].unique()[0]

            curr_os_venv_sub_df = curr_os_sub_df[curr_os_sub_df["venv"]==venv]
            editors_N=curr_os_venv_sub_df["editor"].shape[0]
            curr_group = np.zeros((editors_N, curr_os_venv_sub_df["N_rep"].iloc[0]))
            colors_use = []
            colors_use_whiskers = []
            plotlabels_use = []
            for editor_i, editor in enumerate(curr_os_venv_sub_df["editor"]):
                #editor=curr_os_venv_sub_df["editor"][0]
                if test=="registration_test":
                    curr_group[editor_i] = zarr_in[OS+" "+editor+" "+venv+"/"+test]
                else:
                    curr_group[editor_i] = zarr_in[OS+" "+editor+" "+venv+"/"+test][:,plot_axis,-1]
                colors_use.append(colors[editor])
                colors_use_whiskers.append(colors[editor])
                colors_use_whiskers.append(colors[editor])
                plotlabels_use.append(plotlabels[editor])
            
            plt.close(1)
            fig=plt.figure(1, figsize=figsize)
            fig.clf()
            bp = plt.boxplot(curr_group.T, patch_artist=True, showfliers=False, 
                             labels=plotlabels_use, widths=boxwidth)
            for patch, color in zip(bp['boxes'], colors_use):
                patch.set_facecolor(color)
                patch.set_edgecolor(color)
            for median in bp['medians']: 
                median.set(color ='white', linewidth = 0.75)
            for whisker, color in zip(bp['whiskers'], colors_use_whiskers):
                whisker.set(color=color, linewidth = lw_boxplots)
            for cap, color in zip(bp['caps'], colors_use_whiskers):
                cap.set(color=color, linewidth = lw_boxplots)
            
            # statistical test:
            curr_group_df = pd.DataFrame(data=curr_group.T, columns=curr_os_venv_sub_df["editor"]).melt()
            normality = pg.normality(pd.DataFrame(data=curr_group.T, columns=curr_os_venv_sub_df["editor"]))
            if False in normality["normal"][:].values:
                stats = pg.kruskal(curr_group_df, dv="value", between="editor")
            else:
                stats = pg.anova(curr_group_df, dv="value", between="editor")
            if stats["p-unc"].values<0.5:
                if False in normality["normal"][:].values:
                    stats_pw = pg.pairwise_ttests(curr_group_df, dv="value", between="editor",
                                                  padjust="sidak", effsize="eta-square")
                else:
                    stats_pw = pg.pairwise_tukey(curr_group_df, dv="value", between="editor",
                                                 effsize="eta-square")
                    stats_pw["p-corr"] = stats_pw["p-tukey"]
                    
                Sig_runner=0
                for iSig in np.arange(0, len(stats_pw), 1):
                    # iSig=0
                    if stats_pw.loc[iSig, 'p-corr']<0.05:
                        A_i, B_i = stats_pw.loc[iSig, 'A'], stats_pw.loc[iSig, 'B']
                        x1 = list(curr_os_venv_sub_df["editor"][:].values).index(A_i)+1
                        x2 = list(curr_os_venv_sub_df["editor"][:].values).index(B_i)+1

                        running_offset = barOffset * Sig_runner
                        y, h = ylim[1] - statsOffset - running_offset, barOffset / 2
                        plt.plot([x1, x2], [y + h, y + h], lw=2,
                                    color="k", #Group_Styles[group]['Color']
                                    alpha=0.75)

                        cohen_d_String = '$\eta^2$=' + str(-stats_pw.loc[iSig, 'eta-square'].round(2))
                        plt.text(x2 - (x2 - x1) / 2, y + h, cohen_d_String,
                                    ha='center', va='bottom', fontsize=8, fontweight='light',
                                    color='k')
                        Sig_runner += 1
            
            plt.ylabel(ylabel, fontsize=12, fontweight="bold")
            #plt.title(title+" ("+venv+")", fontweight="bold")
            plt.title(venv, fontweight="bold")
            axis = plt.gca()
            axis.spines['right'].set_visible(False)
            axis.spines['top'].set_visible(False)
            axis.spines['bottom'].set_linewidth(2)
            axis.spines['left'].set_linewidth(2)
            axis.xaxis.set_tick_params(width=2, length=8)
            axis.yaxis.set_tick_params(width=2, length=8)
            plt.xlim(0.0,editors_N+1.0)
            plt.xticks(np.arange(1,editors_N+1,1),fontsize=12, fontweight="bold", 
                       rotation=90)
            """ plt.tick_params(axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False) # labels along the bottom edge are off """
            plt.ylim(ylim)
            plt.yticks(yticks,fontsize=12, fontweight="bold")
            plt.tight_layout()
            if cumulative:
                plotname = OS+" "+venv+" "+test+" (cumulative)"
            else:
                plotname = OS+" "+venv+" "+test
            plt.savefig("plots/evaluation/"+plotname+" bp.pdf")
            if show_plot:
                plt.show()

def boxplot_venv_per_os_per_editor(df_in, zarr_in, colors, plotlabels,
                                    cumulative=False, figsize=(5,3.5), lw_boxplots = 2.5,
                                    ylim=(0,1), yticks=np.arange(0,1),title="",ylabel="",
                                    test="exponential_test", show_plot=False,
                                    statsOffset=1, barOffset=0.1, boxwidth=1):
    if cumulative:
        plot_axis = 2
    else:
        plot_axis = 1
    for OS in df_in["OS"].unique():
        #OS = df_in["OS"].unique()[0]
        curr_os_sub_df = df_in[df_in["OS"]==OS]
        for editor in curr_os_sub_df["editor"]:
            #editor = curr_os_sub_df["editor"].unique()[0]
            curr_os_editor_sub_df = curr_os_sub_df[curr_os_sub_df["editor"]==editor]
            venv_N=curr_os_editor_sub_df["venv"].shape[0]
            curr_group = np.zeros((venv_N, curr_os_editor_sub_df["N_rep"].iloc[0]))
            colors_use = []
            colors_use_whiskers = []
            plotlabels_use = []
            for venv_i, venv in enumerate(curr_os_editor_sub_df["venv"]):
                #venv=curr_os_editor_sub_df["venv"].iloc[0]
                if test=="registration_test":
                    curr_group[venv_i] = zarr_in[OS+" "+editor+" "+venv+"/"+test]
                else:
                    curr_group[venv_i] = zarr_in[OS+" "+editor+" "+venv+"/"+test][:,plot_axis,-1]
                colors_use.append(colors[editor])
                colors_use_whiskers.append(colors[editor])
                colors_use_whiskers.append(colors[editor])
                plotlabels_use.append(plotlabels[venv])
            
            plt.close(1)
            fig=plt.figure(1, figsize=figsize)
            fig.clf()
            bp = plt.boxplot(curr_group.T, patch_artist=True, showfliers=False, 
                             labels=plotlabels_use, widths=boxwidth)
            for patch, color in zip(bp['boxes'], colors_use):
                patch.set_facecolor(color)
                patch.set_edgecolor(color)
            for median in bp['medians']: 
                median.set(color ='white', linewidth = 0.75)
            for whisker, color in zip(bp['whiskers'], colors_use_whiskers):
                whisker.set(color=color, linewidth = lw_boxplots)
            for cap, color in zip(bp['caps'], colors_use_whiskers):
                cap.set(color=color, linewidth = lw_boxplots)
            
            # statistical test:
            curr_group_df = pd.DataFrame(data=curr_group.T, columns=curr_os_editor_sub_df["venv"]).melt()
            normality = pg.normality(pd.DataFrame(data=curr_group.T, columns=curr_os_editor_sub_df["venv"]))
            if False in normality["normal"][:].values:
                stats = pg.kruskal(curr_group_df, dv="value", between="venv")
            else:
                stats = pg.anova(curr_group_df, dv="value", between="venv")
            if stats["p-unc"].values<0.5:
                if False in normality["normal"][:].values:
                    stats_pw = pg.pairwise_ttests(curr_group_df, dv="value", between="venv",
                                                  padjust="sidak", effsize="eta-square")
                else:
                    stats_pw = pg.pairwise_tukey(curr_group_df, dv="value", between="venv",
                                                 effsize="eta-square")
                    stats_pw["p-corr"] = stats_pw["p-tukey"]
                    
                Sig_runner=0
                for iSig in np.arange(0, len(stats_pw), 1):
                    # iSig=0
                    if stats_pw.loc[iSig, 'p-corr']<0.05:
                        A_i, B_i = stats_pw.loc[iSig, 'A'], stats_pw.loc[iSig, 'B']
                        x1 = list(curr_os_editor_sub_df["venv"][:].values).index(A_i)+1
                        x2 = list(curr_os_editor_sub_df["venv"][:].values).index(B_i)+1

                        running_offset = barOffset * Sig_runner
                        y, h = ylim[1] - statsOffset - running_offset, barOffset / 2
                        plt.plot([x1, x2], [y + h, y + h], lw=2,
                                    color="k", #Group_Styles[group]['Color']
                                    alpha=0.75)

                        cohen_d_String = '$\eta^2$=' + str(-stats_pw.loc[iSig, 'eta-square'].round(2))
                        plt.text(x2 - (x2 - x1) / 2, y + h, cohen_d_String,
                                    ha='center', va='bottom', fontsize=8, fontweight='light',
                                    color='k')
                        Sig_runner += 1
            
            plt.ylabel(ylabel, fontsize=12, fontweight="bold")
            #plt.title(title+" ("+venv+")", fontweight="bold")
            plt.title(editor, fontweight="bold")
            axis = plt.gca()
            axis.spines['right'].set_visible(False)
            axis.spines['top'].set_visible(False)
            axis.spines['bottom'].set_linewidth(2)
            axis.spines['left'].set_linewidth(2)
            axis.xaxis.set_tick_params(width=2, length=8)
            axis.yaxis.set_tick_params(width=2, length=8)
            plt.xlim(0.0,venv_N+1.0)
            plt.xticks(np.arange(1,venv_N+1,1),fontsize=12, fontweight="bold", 
                       rotation=90)
            """ plt.tick_params(axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False) # labels along the bottom edge are off """
            plt.ylim(ylim)
            plt.yticks(yticks,fontsize=12, fontweight="bold")
            plt.tight_layout()
            if cumulative:
                plotname = OS+" "+editor+" "+test+" (cumulative)"
            else:
                plotname = OS+" "+editor+" "+test
            plt.savefig("plots/evaluation/"+plotname+" bp.pdf")
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
                        title=f"Average cumulative time for calculating $e^N$ ({N_rep} reps)\n",
                        test="exponential_test", 
                        colors=colors, linestyles=linestyles, round_precision=2)
plot_multiple_1D_series(df_in=zarr_df, zarr_in=zarr_in, figsize=(6,4.5),
                        cumulative=False, 
                        xlim_add=5, xticks_steps=20000,
                        ylim=(0,0.000350), yticks=np.arange(0,0.000351,0.000030),
                        title=f"Average time for calculating $e^N$ ({N_rep} reps)",
                        test="exponential_test", 
                        colors=colors, linestyles=linestyles, round_precision=2)
boxplot_editors_per_os_per_venv(df_in=zarr_df, zarr_in=zarr_in, figsize=(3,5.5),
                                colors=colors, plotlabels=plotlabels,
                                cumulative=True, lw_boxplots = 1.5,
                                ylim=(0,1.00), yticks=np.arange(0,1.01,0.1),
                                title=f"Average cumulative time for calculating $e^N$ for N={zarr_df.iloc[0]['N exponentiation']}",
                                ylabel=f"cumulative time for $N$={zarr_df.iloc[0]['N exponentiation']} [s]",
                                test="exponential_test", show_plot=False,
                                statsOffset=0.05, barOffset=0.045, boxwidth=0.7)
boxplot_venv_per_os_per_editor(df_in=zarr_df, zarr_in=zarr_in, figsize=(3,5.5),
                                colors=colors, plotlabels=plotlabels,
                                cumulative=True, lw_boxplots = 1.5,
                                ylim=(0,1.00), yticks=np.arange(0,1.01,0.1),
                                title=f"Average cumulative time for calculating $e^N$ for N={zarr_df.iloc[0]['N exponentiation']}",
                                ylabel=f"cumulative time for $N$={zarr_df.iloc[0]['N exponentiation']} [s]",
                                test="exponential_test", show_plot=False,
                                statsOffset=0.05, barOffset=0.045, boxwidth=0.7)
# %% EVALUATE THE TRANSPOSING TEST
plot_multiple_1D_series(df_in=zarr_df, zarr_in=zarr_in, figsize=(6,4.5),
                        cumulative=False, 
                        xlim_add=5, xticks_steps=250,
                        ylim=(0,0.050), yticks=np.arange(0,0.051,0.005),
                        title=f"Average time for allocating and\ntransposing an $N \\times N$ array ({N_rep} reps)\n",
                        test="transpose_test", 
                        colors=colors, linestyles=linestyles, round_precision=2)
boxplot_editors_per_os_per_venv(df_in=zarr_df, zarr_in=zarr_in, figsize=(3,5.5),
                                colors=colors, plotlabels=plotlabels,
                                cumulative=False, lw_boxplots = 1.5,
                                ylim=(0,0.075), yticks=np.arange(0,0.076,0.005),
                                title=f"Average time for allocating and\ntransposing an $N \\times N$ array for N={zarr_df.iloc[0]['N transpose']}",
                                ylabel=f"time for $N$={zarr_df.iloc[0]['N transpose']} [s]",
                                test="transpose_test", show_plot=False,
                                statsOffset=0.0025, barOffset=0.0035, boxwidth=0.7)
boxplot_venv_per_os_per_editor(df_in=zarr_df, zarr_in=zarr_in, figsize=(3,5.5),
                                colors=colors, plotlabels=plotlabels,
                                cumulative=False, lw_boxplots = 1.5,
                                ylim=(0,0.075), yticks=np.arange(0,0.076,0.005),
                                title=f"Average time for allocating and\ntransposing an $N \\times N$ array for N={zarr_df.iloc[0]['N transpose']}",
                                ylabel=f"time for $N$={zarr_df.iloc[0]['N transpose']} [s]",
                                test="transpose_test", show_plot=False,
                                statsOffset=0.0025, barOffset=0.0035, boxwidth=0.7)
# %% EVALUATE THE SVD TEST
plot_multiple_1D_series(df_in=zarr_df, zarr_in=zarr_in, figsize=(6,4.5),
                        cumulative=False, 
                        xlim_add=5, xticks_steps=50,
                        ylim=(0,0.35), yticks=np.arange(0,0.36,0.05),
                        title=f"Average time for SVD of an $N \\times N$ array ({N_rep} reps)\n",
                        test="svd_test", 
                        colors=colors, linestyles=linestyles, round_precision=2)
boxplot_editors_per_os_per_venv(df_in=zarr_df, zarr_in=zarr_in, figsize=(3,5.5),
                                colors=colors, plotlabels=plotlabels,
                                cumulative=False, lw_boxplots = 1.5,
                                ylim=(0,0.40), yticks=np.arange(0,0.41,0.05),
                                title=f"Average time for SVD for N={zarr_df.iloc[0]['N svd']}",
                                ylabel=f"time for $N$={zarr_df.iloc[0]['N svd']} [s]",
                                test="svd_test", show_plot=False,
                                statsOffset=0.025, barOffset=0.02, boxwidth=0.7)
boxplot_venv_per_os_per_editor(df_in=zarr_df, zarr_in=zarr_in, figsize=(3,5.5),
                                colors=colors, plotlabels=plotlabels,
                                cumulative=False, lw_boxplots = 1.5,
                                ylim=(0,0.40), yticks=np.arange(0,0.41,0.05),
                                title=f"Average time for SVD for N={zarr_df.iloc[0]['N svd']}",
                                ylabel=f"time for $N$={zarr_df.iloc[0]['N svd']} [s]",
                                test="svd_test", show_plot=False,
                                statsOffset=0.025, barOffset=0.02, boxwidth=0.7)
# %% EVALUATE THE REGISTRATION TEST
boxplot_editors_per_os_per_venv(df_in=zarr_df, zarr_in=zarr_in, figsize=(3,5.5),
                                colors=colors, plotlabels=plotlabels,
                                cumulative=False, lw_boxplots = 1.5,
                                ylim=(40,190), yticks=np.arange(0,181,10),
                                title=f"Average time for image\nregistration task ({N_rep} reps)\n",
                                ylabel="time [s]",
                                test="registration_test", show_plot=False,
                                statsOffset=10, barOffset=7.75, boxwidth=0.7)
boxplot_venv_per_os_per_editor(df_in=zarr_df, zarr_in=zarr_in, figsize=(3,5.5),
                                colors=colors, plotlabels=plotlabels,
                                cumulative=False, lw_boxplots = 1.5,
                                ylim=(40,190), yticks=np.arange(0,181,10),
                                title=f"Average time for image\nregistration task ({N_rep} reps)",
                                ylabel="time [s]",
                                test="registration_test", show_plot=False,
                                statsOffset=10, barOffset=7.75, boxwidth=0.7)
# %% END


