import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def abs2(data):
    return np.abs(data)**2


def borderless_imshow_save(data, outputname, size=1, dpi=80):
    width = 1 * size
    height = data.shape[0] / data.shape[1] * size
    size = (width, height)
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(data)
    plt.savefig(outputname, dpi=dpi)
    plt.close(fig)


def imshow_intensities(amplitudes, imshow_opts={}, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    intensities = np.abs(amplitudes)**2
    ax.imshow(intensities, interpolation='nearest', cmap='magma',
              origin='lower', **imshow_opts)
    ax.axis('off')


def add_noise_to_array(data, noise_level=0.1):
    """Add white noise to the data."""
    data = np.asarray(data)
    range_ = data.max() - data.min()
    return data + np.random.randn(*data.shape) * (noise_level * range_)

