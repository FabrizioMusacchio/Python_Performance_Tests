"""
A script to evaluate the collected results from my IDE peformance test.

author: Fabrizio Musacchio (fabriziomusacchio.com)
date:   Nov 11, 2022
"""
# %% IMPORTS
import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import zarr
import itertools
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
plotlabels = {"VS Code (interactive)": "VS Code$_\mathregular{i}$", 
              "VS Code (terminal)":    "VS Code$_\mathregular{t}$",
              "PyCharm":    "PyCharm", 
              "Jupyter":    "Jupyter",
              "conda":      "conda",
              "python":     "python",
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
                """ plt.text(curr_array[0,0,-1:], np.mean(curr_array[:,plot_axis,-1:], axis=0),
                         " ⟵ " + str(np.mean(curr_array[:,plot_axis,-1:], axis=0).round(round_precision)[0]) +" s",
                         color=colors[editor],
                         ha="left", va="center", clip_on=False) """
            N=curr_array[0,0,:].shape[0]
            plt.xlabel("N", fontsize=13, fontweight="bold")
            plt.ylabel("time [s]", fontsize=13, fontweight="bold")
            plt.title(title+" "+venv, fontweight="bold", fontsize=13)
            plt.legend(loc="upper left", fontsize=10)
            axis = plt.gca()
            axis.spines['right'].set_visible(False)
            axis.spines['top'].set_visible(False)
            axis.spines['bottom'].set_linewidth(2.5)
            axis.spines['left'].set_linewidth(2.5)
            axis.xaxis.set_tick_params(width=2.5, length=8)
            axis.yaxis.set_tick_params(width=2.5, length=8)
            plt.xlim(0,N+xlim_add)
            plt.xticks(np.arange(0,N+1,xticks_steps),fontsize=13, fontweight="bold")
            plt.ylim(ylim)
            plt.yticks(yticks,fontsize=13, fontweight="bold")
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
                if test=="registration_test" or test=="import_test":
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
            for entry_i in range(curr_group.shape[0]):
                plt.plot(entry_i+1, np.median(curr_group[entry_i]), "o", 
                         markeredgewidth=1.5,
                         markeredgecolor=colors_use[entry_i], markerfacecolor="white")
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

                        # cohen_d_String = '$\eta^2$=' + str(-stats_pw.loc[iSig, 'eta-square'].round(2))
                        # cohen_d_String =  str(-stats_pw.loc[iSig, 'eta-square'].round(2))
                        if np.abs(stats_pw.loc[iSig, 'eta-square'])<=0.35:
                            cohen_d_String = "${⦁}$"
                        elif np.abs(stats_pw.loc[iSig, 'eta-square'])<=0.8:
                            cohen_d_String = "${⦁⦁}$"
                        else:
                            cohen_d_String = "${⦁⦁⦁}$"
                        plt.text(x2 - (x2 - x1) / 2, y + h, cohen_d_String,
                                    ha='center', va='bottom', fontsize=10, fontweight='normal',
                                    color='k', stretch='ultra-condensed', family="arial")
                        Sig_runner += 1
            
            plt.ylabel(ylabel, fontsize=14, fontweight="bold")
            #plt.title(title+" ("+venv+")", fontweight="bold")
            plt.title(venv, fontweight="bold", fontsize=14)
            # plt.title(plotlabels_use[editor_i], fontweight="bold")
            axis = plt.gca()
            axis.spines['right'].set_visible(False)
            axis.spines['top'].set_visible(False)
            axis.spines['bottom'].set_linewidth(2.5)
            axis.spines['left'].set_linewidth(2.5)
            axis.xaxis.set_tick_params(width=2.5, length=8)
            axis.yaxis.set_tick_params(width=2.5, length=8)
            plt.xlim(0.0,editors_N+1.0)
            plt.xticks(np.arange(1,editors_N+1,1),fontsize=14, fontweight="bold", 
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
        #OS = df_in["OS"].unique()[2]
        curr_os_sub_df = df_in[df_in["OS"]==OS]
        for editor_i, editor in enumerate(curr_os_sub_df["editor"].unique()):
            # editor_i=2
            #editor = curr_os_sub_df["editor"].unique()[editor_i]
            curr_os_editor_sub_df = curr_os_sub_df[curr_os_sub_df["editor"]==editor]
            venv_N=curr_os_editor_sub_df["venv"].shape[0]
            curr_group = np.zeros((venv_N, curr_os_editor_sub_df["N_rep"].iloc[0]))
            colors_use = []
            colors_use_whiskers = []
            plotlabels_use = []
            for venv_i, venv in enumerate(curr_os_editor_sub_df["venv"]):
                #venv=curr_os_editor_sub_df["venv"].iloc[0]
                if test=="registration_test" or test=="import_test":
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
            bp = plt.boxplot(curr_group.T, patch_artist=True, showfliers=True, 
                             labels=plotlabels_use, widths=boxwidth)
            for entry_i in range(curr_group.shape[0]):
                plt.plot(entry_i+1, np.median(curr_group[entry_i]), "o", 
                         markeredgewidth=1.5,
                         markeredgecolor=colors_use[entry_i], markerfacecolor="white")
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

                        # cohen_d_String = '$\eta^2$=' + str(-stats_pw.loc[iSig, 'eta-square'].round(2))
                        # cohen_d_String = str(-stats_pw.loc[iSig, 'eta-square'].round(2))
                        if np.abs(stats_pw.loc[iSig, 'eta-square'])<=0.35:
                            cohen_d_String = "${⦁}$"
                        elif np.abs(stats_pw.loc[iSig, 'eta-square'])<=0.8:
                            cohen_d_String = "${⦁⦁}$"
                        else:
                            cohen_d_String = "${⦁⦁⦁}$"
                        plt.text(x2 - (x2 - x1) / 2, y + h, cohen_d_String,
                                    ha='center', va='bottom', fontsize=10, fontweight='normal',
                                    color='k', family="arial")
                        Sig_runner += 1
            
            plt.ylabel(ylabel, fontsize=14, fontweight="bold")
            #plt.title(title+" ("+venv+")", fontweight="bold")
            # plt.title(venv, fontweight="bold")
            plt.title(plotlabels[editor], fontweight="bold", fontsize=14)
            axis = plt.gca()
            axis.spines['right'].set_visible(False)
            axis.spines['top'].set_visible(False)
            axis.spines['bottom'].set_linewidth(2.5)
            axis.spines['left'].set_linewidth(2.5)
            axis.xaxis.set_tick_params(width=2.5, length=8)
            axis.yaxis.set_tick_params(width=2.5, length=8)
            plt.xlim(0.0,venv_N+1.0)
            plt.xticks(np.arange(1,venv_N+1,1),fontsize=14, fontweight="bold", 
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

tests= ["import_test", "exponential_test", "transpose_test", "svd_test", "registration_test"]

def summary_plot(df_in, zarr_in, plotlabels, tests):
    
    collector_df = pd.DataFrame()
    for OS in df_in["OS"].unique():
        #OS = df_in["OS"].unique()[0]
        curr_os_sub_df = df_in[df_in["OS"]==OS]
        for venv in curr_os_sub_df["venv"].unique():
            #venv = curr_os_sub_df["venv"].unique()[0]
            curr_os_venv_sub_df = curr_os_sub_df[curr_os_sub_df["venv"]==venv]
            
            for test in tests:
                editors_N=curr_os_venv_sub_df["editor"].shape[0]
                curr_group = np.zeros((editors_N, curr_os_venv_sub_df["N_rep"].iloc[0]))
                plotlabels_use=[]
                for editor_i, editor in enumerate(curr_os_venv_sub_df["editor"]):
                    #editor=curr_os_venv_sub_df["editor"][0]
                    if test=="exponential_test":
                        plot_axis = 2
                    else:
                        plot_axis = 1
                    if test=="registration_test" or test=="import_test":
                        curr_group[editor_i] = zarr_in[OS+" "+editor+" "+venv+"/"+test]
                    else:
                        curr_group[editor_i] = zarr_in[OS+" "+editor+" "+venv+"/"+test][:,plot_axis,-1]
                    plotlabels_use.append(plotlabels[editor])
                
                collector_tmp_df = pd.DataFrame(index=[0])
                collector_tmp_df["OS"]=OS
                if OS=="macOS":
                    collector_tmp_df["OS_join"]="macOS"
                else:
                    collector_tmp_df["OS_join"]="WinLin"
                collector_tmp_df["test"]=test
                collector_tmp_df["venv"]=venv
                for plotlabel_i, plotlabel in enumerate(plotlabels_use):
                    collector_tmp_df[plotlabel]=np.median(curr_group, axis=1)[plotlabel_i]
                collector_df = pd.concat([collector_df, collector_tmp_df], ignore_index=True)
    
    # normalize via z-scoring:
    collector_norm_df = pd.DataFrame()
    OS_join_list =["macOS", "WinLin"]
    for OS in OS_join_list:
        curr_os_collector_df = collector_df[collector_df["OS_join"]==OS]
        for test in tests:
            curr_os_test_collector_df = curr_os_collector_df[curr_os_collector_df["test"]==test]
            mu    = np.mean(curr_os_test_collector_df[plotlabels_use].values)
            sigma = np.std(curr_os_test_collector_df[plotlabels_use].values)
            collector_norm_tmp_df = curr_os_test_collector_df.copy()
            collector_norm_tmp_df[plotlabels_use] = (collector_norm_tmp_df[plotlabels_use]-mu)/sigma
            collector_norm_tmp_df["mu"] = mu
            collector_norm_tmp_df["sigma"] = sigma
            collector_norm_df = pd.concat([collector_norm_df, collector_norm_tmp_df], ignore_index=True)
    array_max = np.max(collector_norm_df[plotlabels_use].values)
    array_min = np.min(collector_norm_df[plotlabels_use].values)
    array_max = np.max(np.abs([array_max, array_min]))
    array_min = -array_max
    
    collector_df.to_csv("plots/evaluation/summary_table (Windows and Linux pooled).csv")
    collector_norm_df.to_csv("plots/evaluation/summary_table_z-scores  (Windows and Linux pooled).csv")
    
    # summary plots for macOS vs WinLin:
    for OS in OS_join_list:
        #OS=OS_join_list[0]
        collector_norm_os_df = collector_norm_df[collector_norm_df["OS_join"]==OS]
        curr_norm_array = []
        for test in tests:
            #test=tests[0]
            collector_norm_os_test_df = collector_norm_os_df[collector_norm_os_df["test"]==test]
            # curr_norm_array.append(collector_norm_os_test_df[plotlabels_use].values.flatten())
            for curr_OS in np.flip(collector_norm_os_test_df["OS"].unique()):
                #curr_OS = collector_norm_os_test_df["OS"].unique()[0]
                collector_norm_subos_test_df = collector_norm_os_test_df[collector_norm_os_test_df["OS"]==curr_OS]
                curr_norm_array.append(collector_norm_subos_test_df[plotlabels_use].values.flatten())
    
        orig_map=plt.cm.get_cmap('coolwarm')
        reversed_map = orig_map #orig_map.reversed()
        ## %%
        
        if OS==OS_join_list[0]:
            k=1
            j=1.5
            OS_shorts = ["M"]*5
            y2_offset_corr = 0.24
        else:
            k=2
            j=2
            OS_shorts = ["W","L"]*5
            y2_offset_corr = 0.5
        line_y_end=5
        line_y_end*=k
        plt.close()
        #figsize=(6.8,3.0*k)
        figsize=(6.8,3.0*j)
        fig=plt.figure(1, figsize=figsize)
        fig.clf()
        plt.pcolor(curr_norm_array, edgecolors='w', linewidths=4, cmap=reversed_map,
                   vmin=array_min, vmax=array_max)
        plt.plot([4,4], [0,line_y_end], '-', c='k')
        plt.plot([8,8], [0,line_y_end], '-', c='k')
        #cbar = plt.colorbar(aspect=10)
        cbar = plt.colorbar(orientation="horizontal", pad=0.05)
        cbar.set_label(label="z-score",weight='bold', fontsize=14)
        cbar.ax.set_xticklabels(np.arange(np.ceil(array_min)-1, np.floor(array_max)+1, 1), fontsize=11, weight='bold')
        cbar.outline.set_visible(False)
        plt.xticks(np.arange(0.5,12))
        plt.yticks(np.arange(0.5,line_y_end))
        plt.text(2,-3.0, "conda", fontsize=14, fontweight="bold", ha='center', va='center')
        plt.text(2+4,-3.0, "python", fontsize=14, fontweight="bold", ha='center', va='center')
        plt.text(2+8,-3.0, "virtualenv", fontsize=14, fontweight="bold", ha='center', va='center')
        axis = plt.gca()
        listOfLists = [list(itertools.repeat(element, k)) for element in tests]
        tests_new = list(itertools.chain.from_iterable(listOfLists))
        axis.set_yticklabels(tests_new, fontsize=14, fontweight="bold")
        axis.set_xticklabels(plotlabels_use*3, rotation=90, fontsize=12, fontweight="bold")
        axis.xaxis.tick_top()
        axis.invert_yaxis()
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.xaxis.set_ticks_position('none')
        axis.yaxis.set_ticks_position('none')
        
        axis2 = axis.twinx()
        # axis2.plot([8,8], [0.5,line_y_end-0.5], '-', c='k')
        axis2.plot([8,8], [y2_offset_corr,line_y_end-y2_offset_corr], '-', c='k')
        axis2.set_yticks(np.arange(0.5,line_y_end+0.5))
        axis2.set_yticklabels(OS_shorts, fontsize=12, fontweight="bold")
        axis2.invert_yaxis()
        axis2.spines['right'].set_visible(False)
        axis2.spines['top'].set_visible(False)
        axis2.spines['bottom'].set_visible(False)
        axis2.spines['left'].set_visible(False)
        axis2.xaxis.set_ticks_position('none')
        axis2.yaxis.set_ticks_position('none')
        axis2.yaxis.set_tick_params(width=0, length=-3)
        axis.yaxis.set_ticks_position('none')
        
        fig.tight_layout()
        plt.savefig("plots/evaluation/summary_table_"+OS+".pdf")
    
    
    # repeat everything, now w/o pooling Windows and Linux in the normalization
    # normalize via z-scoring:
    collector_norm_df = pd.DataFrame()
    OS_join_list =["macOS", "WinLin"]
    for OS in df_in["OS"].unique():
        curr_os_collector_df = collector_df[collector_df["OS"]==OS]
        for test in tests:
            curr_os_test_collector_df = curr_os_collector_df[curr_os_collector_df["test"]==test]
            mu    = np.mean(curr_os_test_collector_df[plotlabels_use].values)
            sigma = np.std(curr_os_test_collector_df[plotlabels_use].values)
            collector_norm_tmp_df = curr_os_test_collector_df.copy()
            collector_norm_tmp_df[plotlabels_use] = (collector_norm_tmp_df[plotlabels_use]-mu)/sigma
            collector_norm_tmp_df["mu"] = mu
            collector_norm_tmp_df["sigma"] = sigma
            collector_norm_df = pd.concat([collector_norm_df, collector_norm_tmp_df], ignore_index=True)
    array_max = np.max(collector_norm_df[plotlabels_use].values)
    array_min = np.min(collector_norm_df[plotlabels_use].values)
    array_max = np.max(np.abs([array_max, array_min]))
    array_min = -array_max
    
    collector_df.to_csv("plots/evaluation/summary_table.csv")
    collector_norm_df.to_csv("plots/evaluation/summary_table_z-scores.csv")
    
    # summary plots for macOS vs Windows vs Linux:
    for OS in df_in["OS"].unique():
        #OS=df_in["OS"].unique()[0]
        collector_norm_os_df = collector_norm_df[collector_norm_df["OS"]==OS]
        curr_norm_array = []
        for test in tests:
            #test=tests[0]
            collector_norm_os_test_df = collector_norm_os_df[collector_norm_os_df["test"]==test]
            # curr_norm_array.append(collector_norm_os_test_df[plotlabels_use].values.flatten())
            for curr_OS in np.flip(collector_norm_os_test_df["OS"].unique()):
                #curr_OS = collector_norm_os_test_df["OS"].unique()[0]
                collector_norm_subos_test_df = collector_norm_os_test_df[collector_norm_os_test_df["OS"]==curr_OS]
                curr_norm_array.append(collector_norm_subos_test_df[plotlabels_use].values.flatten())
    
        orig_map=plt.cm.get_cmap('coolwarm')
        reversed_map = orig_map #orig_map.reversed()
        ## %%
        
        if OS=="macOS":
            OS_shorts = ["M"]*5
        elif OS=="Linux":
            OS_shorts = ["L"]*5
        elif OS=="Windows":
            OS_shorts = ["W"]*5
        
        k=1
        j=1.5
        y2_offset_corr = 0.24
        
        line_y_end=5
        line_y_end*=k
        plt.close()
        #figsize=(6.8,3.0*k)
        figsize=(6.8,3.0*j)
        fig=plt.figure(1, figsize=figsize)
        fig.clf()
        plt.pcolor(curr_norm_array, edgecolors='w', linewidths=4, cmap=reversed_map,
                   vmin=array_min, vmax=array_max)
        plt.plot([4,4], [0,line_y_end], '-', c='k')
        plt.plot([8,8], [0,line_y_end], '-', c='k')
        #cbar = plt.colorbar(aspect=10)
        cbar = plt.colorbar(orientation="horizontal", pad=0.05)
        cbar.set_label(label="z-score",weight='bold', fontsize=14)
        cbar.ax.set_xticklabels(np.arange(np.ceil(array_min)-1, np.floor(array_max)+1, 1), fontsize=11, weight='bold')
        cbar.outline.set_visible(False)
        plt.xticks(np.arange(0.5,12))
        plt.yticks(np.arange(0.5,line_y_end))
        plt.text(2,-3.0, "conda", fontsize=14, fontweight="bold", ha='center', va='center')
        plt.text(2+4,-3.0, "python", fontsize=14, fontweight="bold", ha='center', va='center')
        plt.text(2+8,-3.0, "virtualenv", fontsize=14, fontweight="bold", ha='center', va='center')
        axis = plt.gca()
        listOfLists = [list(itertools.repeat(element, k)) for element in tests]
        tests_new = list(itertools.chain.from_iterable(listOfLists))
        axis.set_yticklabels(tests_new, fontsize=14, fontweight="bold")
        axis.set_xticklabels(plotlabels_use*3, rotation=90, fontsize=12, fontweight="bold")
        axis.xaxis.tick_top()
        axis.invert_yaxis()
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.xaxis.set_ticks_position('none')
        axis.yaxis.set_ticks_position('none')
        
        axis2 = axis.twinx()
        # axis2.plot([8,8], [0.5,line_y_end-0.5], '-', c='k')
        axis2.plot([8,8], [y2_offset_corr,line_y_end-y2_offset_corr], '-', c='k')
        axis2.set_yticks(np.arange(0.5,line_y_end+0.5))
        axis2.set_yticklabels(OS_shorts, fontsize=12, fontweight="bold")
        axis2.invert_yaxis()
        axis2.spines['right'].set_visible(False)
        axis2.spines['top'].set_visible(False)
        axis2.spines['bottom'].set_visible(False)
        axis2.spines['left'].set_visible(False)
        axis2.xaxis.set_ticks_position('none')
        axis2.yaxis.set_ticks_position('none')
        axis2.yaxis.set_tick_params(width=0, length=-3)
        axis.yaxis.set_ticks_position('none')
        
        fig.tight_layout()
        plt.savefig("plots/evaluation/summary_table_single_"+OS+".pdf")
    
        ## %%

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
plot_multiple_1D_series(df_in=zarr_df, zarr_in=zarr_in, figsize=(3.6,4.5),
                        cumulative=True, 
                        xlim_add=5, xticks_steps=30000,
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
boxplot_editors_per_os_per_venv(df_in=zarr_df, zarr_in=zarr_in, figsize=(2.25,5.5),
                                colors=colors, plotlabels=plotlabels,
                                cumulative=True, lw_boxplots = 1.5,
                                ylim=(0,1.00), yticks=np.arange(0,1.01,0.1),
                                title=f"Average cumulative time for calculating $e^N$ for N={zarr_df.iloc[0]['N exponentiation']}",
                                ylabel=f"cumulative time for $N$={zarr_df.iloc[0]['N exponentiation']} [s]",
                                test="exponential_test", show_plot=False,
                                statsOffset=0.05, barOffset=0.045, boxwidth=0.9)
boxplot_venv_per_os_per_editor(df_in=zarr_df, zarr_in=zarr_in, figsize=(2.10,5.5),
                                colors=colors, plotlabels=plotlabels,
                                cumulative=True, lw_boxplots = 1.5,
                                ylim=(0,1.00), yticks=np.arange(0,1.01,0.1),
                                title=f"Average cumulative time for calculating $e^N$ for N={zarr_df.iloc[0]['N exponentiation']}",
                                ylabel=f"cumulative time for $N$={zarr_df.iloc[0]['N exponentiation']} [s]",
                                test="exponential_test", show_plot=False,
                                statsOffset=0.05, barOffset=0.045, boxwidth=0.9)
# %% EVALUATE THE TRANSPOSING TEST
plot_multiple_1D_series(df_in=zarr_df, zarr_in=zarr_in, figsize=(3.8,4.5),
                        cumulative=False, 
                        xlim_add=5, xticks_steps=500,
                        ylim=(0,0.050), yticks=np.arange(0,0.051,0.005),
                        title=f"Average time for allocating and\ntransposing an $N \\times N$ array ({N_rep} reps)\n",
                        test="transpose_test", 
                        colors=colors, linestyles=linestyles, round_precision=2)
boxplot_editors_per_os_per_venv(df_in=zarr_df, zarr_in=zarr_in, figsize=(2.45,5.5),
                                colors=colors, plotlabels=plotlabels,
                                cumulative=False, lw_boxplots = 1.5,
                                ylim=(0,0.075), yticks=np.arange(0,0.076,0.005),
                                title=f"Average time for allocating and\ntransposing an $N \\times N$ array for N={zarr_df.iloc[0]['N transpose']}",
                                ylabel=f"time for $N$={zarr_df.iloc[0]['N transpose']} [s]",
                                test="transpose_test", show_plot=False,
                                statsOffset=0.00425, barOffset=0.0035, boxwidth=0.9)
boxplot_venv_per_os_per_editor(df_in=zarr_df, zarr_in=zarr_in, figsize=(2.3,5.5),
                                colors=colors, plotlabels=plotlabels,
                                cumulative=False, lw_boxplots = 1.5,
                                ylim=(0,0.075), yticks=np.arange(0,0.076,0.005),
                                title=f"Average time for allocating and\ntransposing an $N \\times N$ array for N={zarr_df.iloc[0]['N transpose']}",
                                ylabel=f"time for $N$={zarr_df.iloc[0]['N transpose']} [s]",
                                test="transpose_test", show_plot=False,
                                statsOffset=0.00425, barOffset=0.0035, boxwidth=0.9)
# %% EVALUATE THE SVD TEST
plot_multiple_1D_series(df_in=zarr_df, zarr_in=zarr_in, figsize=(3.7,4.5),
                        cumulative=False, 
                        xlim_add=0, xticks_steps=100,
                        ylim=(0,0.35), yticks=np.arange(0,0.36,0.05),
                        title=f"Average time for SVD of an $N \\times N$ array\n",
                        test="svd_test", 
                        colors=colors, linestyles=linestyles, round_precision=2)
boxplot_editors_per_os_per_venv(df_in=zarr_df, zarr_in=zarr_in, figsize=(2.45,5.5),
                                colors=colors, plotlabels=plotlabels,
                                cumulative=False, lw_boxplots = 1.5,
                                ylim=(0,0.40), yticks=np.arange(0,0.41,0.05),
                                title=f"Average time for SVD for N={zarr_df.iloc[0]['N svd']}",
                                ylabel=f"time for $N$={zarr_df.iloc[0]['N svd']} [s]",
                                test="svd_test", show_plot=False,
                                statsOffset=0.025, barOffset=0.02, boxwidth=0.9)
boxplot_venv_per_os_per_editor(df_in=zarr_df, zarr_in=zarr_in, figsize=(2.2,5.5),
                                colors=colors, plotlabels=plotlabels,
                                cumulative=False, lw_boxplots = 1.5,
                                ylim=(0,0.40), yticks=np.arange(0,0.41,0.05),
                                title=f"Average time for SVD for N={zarr_df.iloc[0]['N svd']}",
                                ylabel=f"time for $N$={zarr_df.iloc[0]['N svd']} [s]",
                                test="svd_test", show_plot=False,
                                statsOffset=0.025, barOffset=0.02, boxwidth=0.9)
# %% EVALUATE THE REGISTRATION TEST
boxplot_editors_per_os_per_venv(df_in=zarr_df, zarr_in=zarr_in, figsize=(2.3,5.5),
                                colors=colors, plotlabels=plotlabels,
                                cumulative=False, lw_boxplots = 1.5,
                                ylim=(40,190), yticks=np.arange(0,181,15),
                                title=f"Average time for image\nregistration task ({N_rep} reps)\n",
                                ylabel="time [s]",
                                test="registration_test", show_plot=False,
                                statsOffset=10, barOffset=7.75, boxwidth=0.9)
boxplot_venv_per_os_per_editor(df_in=zarr_df, zarr_in=zarr_in, figsize=(2.15,5.5),
                                colors=colors, plotlabels=plotlabels,
                                cumulative=False, lw_boxplots = 1.5,
                                ylim=(40,190), yticks=np.arange(0,181,15),
                                title=f"Average time for image\nregistration task ({N_rep} reps)",
                                ylabel="time [s]",
                                test="registration_test", show_plot=False,
                                statsOffset=10, barOffset=7.75, boxwidth=0.9)
# %% EVALUATE THE IMPORTS TEST
boxplot_editors_per_os_per_venv(df_in=zarr_df, zarr_in=zarr_in, figsize=(2.25,5.5),
                                colors=colors, plotlabels=plotlabels,
                                cumulative=False, lw_boxplots = 1.5,
                                ylim=(0,25.5), yticks=np.arange(0,25.1,2),
                                title=f"Average time for imports ({N_rep} reps)\n",
                                ylabel="time [s]",
                                test="import_test", show_plot=False,
                                statsOffset=1.65, barOffset=1.10, boxwidth=0.9)
boxplot_venv_per_os_per_editor(df_in=zarr_df, zarr_in=zarr_in, figsize=(2.0,5.5),
                                colors=colors, plotlabels=plotlabels,
                                cumulative=False, lw_boxplots = 1.5,
                                ylim=(0,25.5), yticks=np.arange(0,25.1,2),
                                title=f"Average time for imports ({N_rep} reps)",
                                ylabel="time [s]",
                                test="import_test", show_plot=False,
                                statsOffset=1.65, barOffset=1.10, boxwidth=0.9)
# %% SUMMARY PLOTS
tests= ["import_test", "exponential_test", "transpose_test", "svd_test", "registration_test"]
summary_plot(df_in=zarr_df, zarr_in=zarr_in, plotlabels=plotlabels, tests=tests)
# %% END


