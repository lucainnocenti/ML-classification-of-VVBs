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


def imshow_stokes_probs(prob_vectors, imshow_opts={}, axs=None, show_axis=True):
    """Plot stokes probabilities associated with a VVB state.

    Attributes
    ----------
    prob_vectors : list of three probability vectors
        Should be of shape (3, N, M). If it is (N, M, 3) transpose is applied.
    """
    if prob_vectors.shape[2] == 3:
        prob_vectors = np.transpose(prob_vectors, (2, 0, 1))
    if axs is None:
        _, axs = plt.subplots(1, ncols=3, sharey=True)
    labels = ['0/1', '+/-', 'L/R']
    for ax, prob_vector, label in zip(axs, prob_vectors, labels):
        ax.imshow(prob_vector, **imshow_opts)
        ax.set_title(label)
        if not show_axis:
            ax.axis('off')
    return axs


def add_noise_to_array(data, noise_level=0.1):
    """Add white noise to the data."""
    data = np.asarray(data)
    range_ = data.max() - data.min()
    return data + np.random.randn(*data.shape) * (noise_level * range_)


def rescale_array_values(array, range_):
    min_, max_ = range_
    arr_min = array.min()
    arr_max = array.max()
    return min_ + (array - arr_min) / (arr_max - arr_min) * (max_ - min_)


def make_into_rgb_format(array):
    """Make an N x M x 3 array fit for the RGB format."""
    # rescale to that min and max are 0 and 255, respectively
    # NOTE: I am not so sure about the correctness of the rescaling, CHECK
    if array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    return rescale_array_values(array, [0, 255]).astype(np.uint8)


def plot_stokes_probs_as_rbg(stokes_probs, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.imshow(make_into_rgb_format(stokes_probs))
    ax.axis('off')
    return ax


def merge_dict_elements(dict_):
    data = None
    for key in list(dict_):
        if data is None:
            data = dict_[key]
        else:
            data = np.append(data, dict_[key], axis=0)
    return data


def truncate_in_reduced_space(data, trained_pca, num_dimensions_left):
    """Use PCA to reduce the dimension, then truncate and go back.

    Attributes
    ----------
    data : np.array of shape num_samples x feature_size
        The data in the original dimension.
    trained_pca : sklearn.PCA instance
        Used to switch to the reduced space, where we truncate the vector,
        and then used again to switch back to the original dimension.
    num_dimensions_left : int
        All components in dimensions with index greater than this number are
        zeroed out (in the reduced space). Should not be larger than the number
        of dimensions that trained_pca reduces the data to.
    """
    reduced_data = trained_pca.transform(data)
    reduced_data[:, num_dimensions_left:] = 0
    return trained_pca.inverse_transform(reduced_data)
