import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, re

from ProteomicsUtils.LoggerConfig import logger_config
from ProteomicsUtils import FileHandling, StatUtils, CalcUtils, PlotUtils

logger = logger_config(__name__)
logger.info("Import OK")

def scatbar_plot(x_col, y_col, data_frame, hue_name, legend, output):
    fig, ax = plt.subplots()
    ax = sns.barplot(x=x_col, y=y_col, data=data_frame, hue=hue_name, dodge=True,errwidth=1.25,alpha=0.25,ci=None)
    sns.swarmplot(x=x_col, y=y_col, data=data_frame, hue=hue_name, dodge=True)

    # To only label once in legend
    handles, labels = ax.get_legend_handles_labels()
    lgd = plt.legend(handles[0:3], legend, title=hue_name, bbox_to_anchor=(1.26, 1.0))

    # To generate custom error bars
    xcentres=list(np.arange(0, len(data_frame[x_col].unique())))
    delt=0.26
    xneg=[x-delt for x in xcentres]
    xpos=[x+delt for x in xcentres]
    xvals=xneg+xpos+xcentres
    xvals.sort()
    sample_list = data_frame[x_col].unique()
    yvals=data_frame.groupby([x_col,"Hue position"]).mean().T[sample_list].T[y_col]
    yerr=data_frame.groupby([x_col,"Hue position"]).std().T[sample_list].T[y_col]

    (_, caps, _) = ax.errorbar(x=xvals,y=yvals,yerr=yerr,fmt=None,capsize=4,errwidth=1.25,ecolor="black")
    for cap in caps:
        cap.set_markeredgewidth(2)

    plt.ylabel(y_col)
    plt.tight_layout()
    plt.autoscale()
    fig.savefig(output, bbox_extra_artists=(lgd,), bbox_inches='tight')

    return fig
