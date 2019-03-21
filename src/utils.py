import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def abs2(data):
    return np.abs(data)**2


def borderless_imshow_save(data, outputname, size=1, dpi=80, imshow_opts={}):
    width = 1 * size
    height = data.shape[0] / data.shape[1] * size
    size = (width, height)
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(data, **imshow_opts)
    plt.savefig(outputname, dpi=dpi)
    plt.close(fig)


def imshow_intensities(amplitudes, imshow_opts={}, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    intensities = np.abs(amplitudes)**2
    ax.imshow(intensities, interpolation='nearest', cmap='magma',
              origin='lower', **imshow_opts)
    ax.axis('off')


def imshow_stokes_probs(prob_vectors, imshow_opts={}, axs=None):
    if axs is None:
        _, axs = plt.subplots(1, ncols=3)
    labels = ['0/1', '+/-', 'L/R']
    for ax, prob_vector, label in zip(axs, prob_vectors, labels):
        ax.imshow(prob_vector, **imshow_opts)
        ax.set_title(label)


def add_noise_to_array(data, noise_level=0.1):
    """Add white noise to the data."""
    data = np.asarray(data)
    range_ = data.max() - data.min()
    return data + np.random.randn(*data.shape) * (noise_level * range_)


def make_into_rgb_format(array):
    """Make an N x M x 3 array fit for the RGB format."""
    # rescale to that min and max are 0 and 255, respectively
    # NOTE: I am not so sure about the correctness of the rescaling, CHECK
    array = array + np.abs(array.min())
    array = (array * 255 / array.max()).astype(np.uint8)
    return array


def merge_dict_elements(dict_):
    data = None
    for key in list(dict_):
        if data is None:
            data = dict_[key]
        else:
            data = np.append(data, dict_[key], axis=0)
    return data
