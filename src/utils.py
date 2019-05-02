import os
import sys
import collections

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


def imshow_intensities(amplitudes=None, intensities=None, imshow_opts={}, ax=None):
    if amplitudes is None and intensities is None:
        raise ValueError('One of `amplitudes` and `intensities` must be given.')
    if amplitudes is not None and intensities is not None:
        raise ValueError('Only one of `amplitudes` and `intensities` must be given.')
    if amplitudes is not None:
        intensities = np.abs(amplitudes)**2
    if ax is None:
        _, ax = plt.subplots(1, 1)
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


def rescale_array_values(array, range_, old_range=None):
    min_, max_ = range_
    if old_range is None:
        old_min = array.min()
        old_max = array.max()
    else:
        old_min, old_max = old_range
    return min_ + (array - old_min) / (old_max - old_min) * (max_ - min_)


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
    """Merge all elements of the dictionary into a single array."""
    data = None
    for key in list(dict_):
        if data is None:
            data = dict_[key]
        else:
            data = np.vstack((data, dict_[key]))
    return data


def dict_of_arrays_to_labeled_array(dict_):
    """Convert dict of arrays into a single array plus an array of labels.

    This function assumes that the keys of the given dictionary are of the form
    `cXX`, where `c` is a single char, and `XX` some integer number.


    """
    data = None
    labels = None
    for key in dict_:
        new_data = dict_[key]
        new_labels = np.full(shape=dict_[key].shape[0], fill_value=key)
        if data is None:
            data = new_data
            labels = new_labels
        else:
            data = np.vstack((data, new_data))
            labels = np.append(labels, new_labels)
    return data, labels


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


def compute_accuracies_per_label(true_labels, predicted_labels):
    possible_labels = list(set(true_labels))  # do not have to be numeric
    # convert the labels into integers to ease handling
    labels_indices = list(range(len(possible_labels)))
    
    accuracies_per_label = collections.OrderedDict()
    for label_idx in labels_indices:
        # we want to compute the output accuracies for this specific label
        accuracies = np.zeros(shape=(len(possible_labels),))
        # extract the elements corresponding to the currently considered true label
        true_labels_indices = np.where(true_labels == possible_labels[label_idx])
        true_labels_per_class = true_labels[true_labels_indices]
        predictions = predicted_labels[true_labels_indices]
        # we iterate over all the labels that the classifier associated with the
        # currently considered true label
        for predicted_label_name, count in collections.Counter(predictions).items():
            # extract the index associated with this predicted label
            predicted_label_idx = possible_labels.index(predicted_label_name)
            # put the number of times this label was predicted where appropriate in the `accuracies` array
            accuracies[predicted_label_idx] = count / len(predictions)
        accuracies_per_label[label_idx] = accuracies
    return accuracies_per_label
