import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, re

from loguru import logger

logger.info("Import OK")

def scatbar_plot(x_col, y_col, plotting_dfs, hue_col, group_col):
    dimensions = len(plotting_dfs)
    fig, axes = plt.subplots(dimensions, 1, figsize=(8, dimensions*3), squeeze=False)

    for x, df in enumerate(plotting_dfs):
        # Generate figures
        br = sns.barplot(x=x_col, y=y_col, data=df, hue=hue_col, dodge=True,errwidth=1.25,alpha=0.25,ci=None, ax=axes[0, x])
        scat = sns.swarmplot(x=x_col, y=y_col, data=df, hue=hue_col, dodge=True, ax=axes[0, x])

        # To generate custom error bars
        sample_list = list(set(df[x_col]))
        number_groups = len(list(set(df[hue_col])))

        bars = br.patches
        xvals = [(bar.get_x() + bar.get_width()/2) for bar in bars]
        xvals.sort()
        yvals=df.groupby([x_col, hue_col]).mean().T[sample_list].T[y_col]
        yerr=df.groupby([x_col, hue_col]).std().T[sample_list].T[y_col]

        (_, caps, _) = axes[0, x].errorbar(x=xvals,y=yvals,yerr=yerr,capsize=4,elinewidth=1.25,ecolor="black", linewidth=0)
        for cap in caps:
            cap.set_markeredgewidth(2)
        axes[0, x].set_ylabel(y_col)

        # To only label once in legend
        handles, labels = axes[0, x].get_legend_handles_labels()
        axes[0, x].legend(handles[0:number_groups], labels[0:number_groups], bbox_to_anchor=(1.26, 1.05), title=hue_col)

        # rotate tick labels

        for label in axes[0, x].get_xticklabels():
            label.set_rotation(45)

    plt.ylabel(y_col)

    for x in range(len(axes)):
        axes[0, x].set_xlabel(x_col)

    plt.tight_layout()
    plt.autoscale()

    return fig, axes
