"""
@uthor: Himaghna Bhattacharjee
Description: Collection of plotting functions
"""

import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from seaborn import heatmap, kdeplot


def plot_parity(x, y, **kwargs):
    """Plot parity plot of x vs y

    """
    plot_params = {
        'alpha': 0.7,
        's': 10,
        'c': 'green',
    }
    if kwargs is not None:
        plot_params.update(kwargs)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.scatter(
        x=x,
        y=y,
        alpha=plot_params['alpha'],
        s=plot_params['s'],
        c=plot_params['c'])
    max_entry = max(max(x), max(y)) + plot_params.get('offset', 5)
    min_entry = min(min(x), min(y))  - plot_params.get('offset', 5)
    axes = plt.gca()
    axes.set_xlim([min_entry, max_entry])
    axes.set_ylim([min_entry, max_entry])
    plt.plot([min_entry, max_entry], [min_entry, max_entry],
             color=plot_params.get('linecolor', 'black'))
    plt.title(plot_params.get('title', ''),
        fontsize=plot_params.get('title_fontsize', 24))
    plt.xlabel(plot_params.get('xlabel', ''),
                   fontsize=plot_params.get('xlabel_fontsize', 20))
    plt.ylabel(plot_params.get('ylabel', ''),
                   fontsize=plot_params.get('ylabel_fontsize', 20))
    plt.xticks(fontsize=plot_params.get('xticksize',24))
    plt.yticks(fontsize=plot_params.get('yticksize',24))
    plt.text(
        plot_params.pop('txt_x', 0.05), plot_params.pop('txt_y', 0.9),
        plot_params.pop('text', None), transform=axes.transAxes,
        fontsize=plot_params.pop('text_fontsize', 16))
    if not plot_params.get('show_plot', True):
        return axes
    plt.tight_layout()
    plt.show()


def plot_density(plot_vector, **kwargs):
    """Plot the similarity density"""

    # get params
    bw = float(kwargs.get('bw', 0.01))
    plt.rcParams['svg.fonttype'] = 'none'
    plot_label = kwargs.get('label', None)
    if plot_label is not None:
        kdeplot(
            plot_vector, shade=kwargs.get('shade', True),
            color=kwargs.get('color', 'orange'), bw=bw, label=plot_label)
        plt.legend(prop={'size': kwargs.get('legend_fontsize', 20)})
    else:
        kdeplot(
            plot_vector, shade=kwargs.get('shade', True),
            color=kwargs.get('color', 'orange'), bw=bw)
    plt.xlabel(kwargs.get('xlabel','Samples'), fontsize=24)
    plt.ylabel(kwargs.get('ylabel','Density'), fontsize=24)
    plt.xticks(fontsize=kwargs.get('xticksize',20))
    plt.yticks(fontsize=kwargs.get('yticksize',20))
    if kwargs.get('title', None) is not None:
        plt.title(title, fontsize=20)


def plot_bivariate(x, y, **kwargs):
    """Plot bivariate distribution of two vectors."""
    plot_params = dict(cmap='Reds', shade=True, shade_lowest=False)
    plot_params.update(kwargs)
    plt.rcParams['svg.fonttype'] = 'none'
    min_entry = plot_params.pop('min_entry',
        min(min(x), min(y))  - plot_params.pop('offset', 5))
    max_entry = plot_params.pop('max_entry',
        max(max(x), max(y)) + plot_params.pop('offset', 5))
    axes = plt.gca()
    axes.set_xlim([min_entry, max_entry])
    axes.set_ylim([min_entry, max_entry])
    plt.plot([min_entry, max_entry], [min_entry, max_entry],
             color=plot_params.pop('linecolor', 'black'))
    plt.title(plot_params.pop('title', ''),
        fontsize=plot_params.pop('title_fontsize', 24))
    plt.xlabel(plot_params.pop('xlabel', ''),
                   fontsize=plot_params.pop('xlabel_fontsize', 24))
    plt.ylabel(plot_params.pop('ylabel', ''),
                   fontsize=plot_params.pop('ylabel_fontsize', 24))
    plt.xticks(fontsize=plot_params.pop('xticksize',20))
    plt.yticks(fontsize=plot_params.pop('yticksize',20))
    plt.text(
        plot_params.pop('txt_x', 0.05), plot_params.pop('txt_y', 0.9),
        plot_params.pop('text', None), transform=axes.transAxes,
        fontsize=plot_params.pop('text_fontsize', 20))
    kdeplot(x, y, ax=axes,**plot_params)


def pretty_plot(x, y, **kwargs):
    """
    Clean plot of y vs x

    Params ::
    x: n x 1 numpy array: values plotted along x axis
    y: n x 1 numpy array: values plotted along y axis

    Returns ::
    if kwargs.show_plot set to False, returns pyplot axis.
    """
    plot_params = {
        'alpha': 0.7,
        'c': 'red',
    }
    if kwargs is not None:
        plot_params.update(kwargs)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.plot(
        x, y, alpha=plot_params['alpha'], c=plot_params['c'],
        marker=plot_params.get('marker', None),
        markerfacecolor=plot_params.get('markerfacecolor', 'ff4893'),
            markeredgecolor=plot_params.get('markeredgecolor', '45454d'),
            markersize=plot_params.get('markersize', 20),
            markeredgewidth=plot_params.get('markeredgewidth', 10))

    plt.title(plot_params.get('title', ''),
        fontsize=plot_params.get('title_fontsize', 24))
    plt.xlabel(plot_params.get('xlabel', ''),
                   fontsize=plot_params.get('xlabel_fontsize', 24))
    plt.ylabel(plot_params.get('ylabel', ''),
                   fontsize=plot_params.get('ylabel_fontsize', 24))
    plt.xticks(fontsize=plot_params.get('xticksize',20))
    plt.yticks(fontsize=plot_params.get('yticksize',20))
    if not plot_params.get('show_plot', True):
        return plt
    plt.tight_layout()
    plt.show()
