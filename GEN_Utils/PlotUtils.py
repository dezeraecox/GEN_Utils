"""
Collection of plotting functions, including histograms and scatterplots in both Matplotlib and Bokeh.
"""

import pandas as pd
import seaborn as sns
import os, re
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import logging
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import brewer
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    PrintfTickFormatter,
    ColorBar,
    glyphs,
    Span
)
from bokeh.io import export_svgs
from bokeh.layouts import gridplot
from GEN_Utils.LoggerConfig import logger_config
from GEN_Utils import FileHandling

logger = logger_config(__name__)
logger.info('Import ok')


def multirow_scatter(dataframe, key, col_head, x_vals, x_label, y_label):
    """
    collects grouped rows of a dataframe as y values using (1) a grouping
    column, then (2) a set column, then plots against provided x values in a
    simple scatter. Plots will be labelled with the group identifier and the
    set names used in legend

    Parameters:
    dataframe: DataFrame
        contains all y-values to be plotted for all groups, in the format
        Group (str), Set (str), Numerical columns (float)
    key: string
        column header used to group the data e.g ProteinID
    col_head: string
        column header used to label each trace for a group e.g. Sequence
    x_vals: list
        predetermined x vals against which each group will be plotted.
        List must have same number of values as the numerical columns to be
        plotted
    x_label, y_label: strings
        labels for the axis of the scatter plot

    Returns:
    fig_dict: Dictionary
        contains fig objects mapped to each group name
    """
    #collect unique keys to use as titles for each graph
    unique_keys = dataframe[key].unique()
    #create empty dict to store figure objects
    fig_dict = {}

    for entry in unique_keys:
        #collect all instances of a single key and set index
        y_cols = dataframe.loc[(dataframe[key] == entry)].set_index(col_head)
        y_cols = y_cols.drop(key, axis=1)
        #collect y_labels
        y_list = y_cols.index.values
        #make empty figure axis
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        #for each x_col, add y_ against urea to plot
        for label in y_list:
            x = x_vals
            y = y_cols.loc[label]
            name = label
            ax.plot(x, y, label=name, marker='.')
        #adjust plot parameters
        Leg_pos = -(len(y_list)/10+0.1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(entry)
        ax.legend(bbox_to_anchor=(0., Leg_pos, 1., .102), loc=4,
               ncol=1, mode="expand", borderaxespad=0.)
        #append figure object to dictionary according to protein name
        fig_dict[entry] = fig

    return fig_dict


def simple_hist(vals, samplename, bins=100, min=None, max=None):
    """
    Generates a simple histogram of the vals list/series provided, over {bins} (default 100) and displays bins from min (default 0) to max (default 5). Returns matplotlib fig object.
    """
    if not min:
        min = vals.min()
    if not max:
        max = vals.max()
    fig = plt.figure()
    plt.hist(vals.dropna(), bins=bins, range=[min, max])
    calc_vals = vals.dropna()
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.title(samplename)
    plt.axvline(np.median(calc_vals),
                color='r', linestyle='dashed', linewidth=1)
    plt.grid(True)

    return fig



def simple_scatter(x,y,title=None, xlabel=None, ylabel=None, colours=None):
    """Generates a simple, non-interactive scatter plot with basic customisation.

    Parameters
    ----------
    x : list/series
        x data points as int or floats
    y : list/series
        corresponding y data points as int or floats
    title : str (default None)
        Title of data being plotted
    xlabel : str (default None)
        Label of x-axis
    ylabel : str (default None)
        Label of y-axis
    colours : list (default None)
        list of colours corresponding to individual datapoints.

    Returns
    -------
    fig
        matplotlib figure object

    """
    fig = plt.scatter(x, y, c = colours, s = 5)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    x_lines = [0, -1, 1]
    y_lines = [1.3]
    for a in range (0,len(x_lines)):
        plt.axvline(x_lines[a], color='gray', linestyle='dashed', linewidth=1)
    for b in range(0,len(y_lines)):
        plt.axhline(y_lines[b], color='gray', linestyle='dashed', linewidth=1) # p-value of 0.05 is considered significant
    plt.grid(True)
    fig = plt.gcf()
    return fig
    #plt.show(fig)


# define the behaviour -> what happens when you pick a dot on the scatterplot by clicking close to it

def annotate(axis, text, x, y):
    """ Worker function for interactive scatterplot"""
    text_annotation = Annotation(text, xy=(x, y), xycoords='data')
    axis.add_artist(text_annotation)

def inter_scatter(xdata,ydata, xlabel, ylabel, colours, title, datalabels):
    """Generates an interactive scatter plot in which clicking on a datapoint labels that point with the datalabel. Also includes definition of "on_pick" function to assist with labelling points.

    Parameters
    ----------
    xdata : list/series
        x data points as int or floats
    ydata : list/series
        corresponding y data points as int or floats
    title : str
        Title of data being plotted
    xlabel : str)
        Label of x-axis
    ylabel : str
        Label of y-axis
    colours : list
        list of colours corresponding to individual datapoints.
    datalabels : list
        list of labels corresponding to individual datapoints.

    Returns
    -------
    fig
        matplotlib figure object

    """
    fig = plt.figure()
    ax = plt.subplot()
    ax.scatter(xdata, ydata, c = colours, picker=True, s = 5 )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    x_lines = [0, -1, 1]
    y_lines = [1.3]
    for a in range (0,len(x_lines)):
        plt.axvline(x_lines[a], color='gray', linestyle='dashed', linewidth=1)
    for b in range(0,len(y_lines)):
        plt.axhline(y_lines[b], color='gray', linestyle='dashed', linewidth=1) # p-value of 0.05 is considered significant
    plt.grid(True)
    datalabels=datalabels
    def onpick(event):
        # step 1: take the index of the dot which was picked
        ind = event.ind
        # step 2: save the actual coordinates of the click, so we can position the text label properly
        label_pos_x = event.mouseevent.xdata
        label_pos_y = event.mouseevent.ydata
        # just in case two dots are very close, this offset will help the labels not appear one on top of each other
        offset = 0
        # if the dots are to close one to another, a list of dots clicked is returned by the matplotlib library
        for i in ind:
            # step 3: take the label for the corresponding instance of the data
            label = datalabels[i]
            # step 4: log it for debugging purposes
            print ("index", i, label)
            # step 5: create and add the text annotation to the scatterplot
            annotate(
                ax,
                label,
                label_pos_x + offset,
                label_pos_y + offset
            )
            # step 6: force re-draw
            ax.figure.canvas.draw_idle()
            # alter the offset just in case there are more than one dots affected by the click
            offset += 0.1
    # connect the click handler function to the scatterplot
    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()
    return fig


def bokeh_scatter_maker(df, x_col, y_col, c_col, title, hover_list, to_svg=False):
    """Generates a Bokeh figure object for the supplied data, with hover interactivity. Points are coloured according to a datacolumn
    Parameters
    ----------
    df : dataframe
        Pandas DataFrame containing the data to be plotted (x, y, and colours, plus optional hover information) as columns.
    x_col : str
        Column name for x_data, also used to label the x-axis
    y_col : str
        Column name for y_data, also used to label the y-axis
    c_col : str
        Column name for data used to set colour of points
    title : str
        Title of figure to be generated
    hover_list : list of tuples
        Each tuples contains the name of the parameter to be shown on hover (e.g. Protein), with the column to be used for mapping (e.g. @{Master Protein Accessions}), both as strings. Column names with spaces should be encased in {}, and map columns preceded with @.
    to_svg : bool, False (default)
        If true, the save button on the generated html plot will save an svg. Be aware this can affect the interactive functions in html version.
    Returns
    -------
    fig
        Bokeh figure object, which can be plotted (using show(fig)) or added to grid layout.
    """

    source = ColumnDataSource(df)
    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

    colors = list(reversed(brewer['Reds'][9]))#brewer['RdYlBu'][25]#["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    mapper = LinearColorMapper(palette=colors, low=df[c_col].min(), high=df[c_col].max())

    fig = figure(title=title, plot_width=500, plot_height=500, tools=TOOLS, toolbar_location='below')

    #creating objects to be added to the figure
    vline = Span(location=0, dimension='height', line_color='red', line_width=1, line_dash='dotted')
    hline = Span(location=0, dimension='width', line_color='grey', line_width=1, line_dash='dotted')
    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
                     ticker=BasicTicker(desired_num_ticks=len(colors)),
                     label_standoff=6, border_line_color=None, location=(0, 0))

    #adding all elements to the figure plot
    fig.renderers.extend([vline, hline])
    fig.grid.grid_line_color = None
    fig.background_fill_color = None
    fig.xaxis.axis_label = str(x_col)
    fig.yaxis.axis_label = str(y_col)
    fig.scatter(x=x_col,
          y=y_col,
          marker='circle', size=15,
          source=source,
          fill_color={'field': c_col, 'transform': mapper},
              line_color="navy", alpha=0.5)
    fig.add_layout(color_bar, 'right')
    fig.select_one(HoverTool).tooltips = hover_list
    #fig.select_one(HoverTool).formatters={'Gene name' : 'printf', 'Ontology' : 'printf',# use 'printf' formatter}
    if to_svg:
        fig.output_backend = "svg"

    return fig



def y_scaler(fig_nums, y_min=None, y_max=None):
    """Replots provided figures with the same y-scale

    Parameters
    ----------
    fig_nums : list
        List of integers refering to the figures to be adjusted i.e. works with the current list of figures defined in matplotlib.pyplot
    y_min : float
        minimum of adjusted y-scale
    y_max : type
        maximum of adjusted y-scale

    Returns
    -------
    scaled_figs : list
        list of figure objects with adjusted y-scale, which can then be saved to pdf/svg or viewed.

    """
    scaled_figs = []
    for x in fig_nums:
        fig = plt.figure(x)
        if not y_min:
            plt.ylim(ymax=y_max)
        if not y_max:
            plt.ylim(ymin=y_min)
        else:
            plt.ylim(ymin=y_min, ymax=y_max)
        scaled_figs.append(fig)
    return scaled_figs


def bokeh_multi_scatter(df, x_col, y_cols, y_label, title, to_svg=False):
    """Generates a Bokeh figure object for the supplied data, with hover interactivity. Points are coloured according to a datacolumn
    Parameters
    ----------
    df : dataframe
        Pandas DataFrame containing the data to be plotted (x, y) as columns.
    x_col : str
        Column name for x_data, also used to label the x-axis
    y_cols : list
        List of columns to be plotted as y data
    y_label : str
        used to label the y-axis
    title : str
        Title of figure to be generated
    to_svg : bool, False (default)
        If true, the save button on the generated html plot will save an svg. Be aware this can affect the interactive functions in html version.
    Returns
    -------
    fig
        Bokeh figure object, which can be plotted (using show(fig)) or added to grid layout.
    """

    source = ColumnDataSource(df)
    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

    fig = figure(title=title, plot_width=500, plot_height=500, tools=TOOLS, toolbar_location='below')

    #adding all elements to the figure plot
    fig.grid.grid_line_color = None
    fig.background_fill_color = None
    fig.xaxis.axis_label = str(x_col)
    fig.yaxis.axis_label = str(y_label)
    for col in y_cols:
        fig.scatter(x=x_col,
          y=col,
          marker='circle', size=10,
          source=source,
          )

    if to_svg:
        fig.output_backend = "svg"

    return fig
