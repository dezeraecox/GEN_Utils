"""
General collection of functions for handling input and output of data, including reading excel files, iterating over folders, saving dataframes to excel and saving figures to pdf or svg formats.
"""

import pandas as pd
import os
import re
from matplotlib.backends.backend_pdf import PdfPages
import xlrd
import logging
from GEN_Utils.LoggerConfig import logger_config

logger = logger_config(__name__)
logger.info('Import ok')


def df_to_excel(output_path, sheetnames, data_frames):
    """Saves list of dataframes to a single excel (xlsx) file.

    Parameters
    ----------
    output_path : str
        Full path to which xlsx file will be saved.
    sheetnames : list of str
        descriptive list of dataframe content, used to label sheets in xlsx file.
    data_frames : list of DataFrames
        DataFrames to be saved. List order must match order of names provided in sheetname.

    Returns
    -------
    None.

    """
    if not output_path.endswith('.xlsx'):
        output_path = output_path+'Results.xlsx'
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    for x in range(0, len(sheetnames)):
        sheetname = sheetnames[x]
        data_frame = data_frames[x]
        data_frame.to_excel(writer, sheet_name=sheetname, index=False)
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


def fig_to_pdf(figs, output_path, fig_type=None):
    """Save matplotlib figure objects to pdf document.

    Parameters
    ----------
    figs : dict or list
        Container with figure objects, either list or dictionary with figure objects as keys.
    output_path : str
        Partial path to folder to which figures will be saved. Figures.pdf is appended internally.
    fig_type : str, optional
        Appended to output_path prior to 'Figures.pdf' if provided.

    Returns
    -------
    None.

    """
    if fig_type:
        output_path = output_path+fig_type+'_'
    if isinstance(figs, dict):
        logger.info('Figure dictionary found')
        figs = list(figs.values())
    logger.info(figs)
    # page manager to allow saving multiple graphs to single pdf

    pdf = PdfPages(output_path + 'Figures.pdf')
    for fig in figs:
        pdf.savefig(fig)
        # fig.clf() #prevent plots from being overlayed onto the first
    # close pdfpages to allow open access
    pdf.close()


def fig_to_svg(fig_names, fig_list, output_path):
    """Save matplotlib figure objects to svg documents.

    Parameters
    ----------
    fig_names : list
        names given to the output svg files in the file path
    fig_list : list
        List of matplotlib figure objects, in order corresponding to fig_names.
    output_path : str
        Partial path to folder to which figures will be saved. Fig_name and extension (.svg) are appended internally.

    Returns
    -------
    None.

    """
    x = 0
    # figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in fig_list:
        figname = fig_names[x]
        filename = output_path + figname
        fig.savefig(filename + '.svg', transparent=True)
        x += 1
