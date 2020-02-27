import math
import os
import sys
import logging

import numpy as np
import pandas as pd
import scipy
import scipy.special
import sklearn
import sklearn.decomposition

# import keras
# import matplotlib.pyplot as plt
# import seaborn as sns
# import tensorflow
from tqdm import tqdm
import progressbar
import collections
import itertools

import src.utils as utils
from src.utils import abs2

# Camera resolution
# camera_width = 1024
# camera_height = 768
camera_width = 300
camera_height = 300
Y, X = np.meshgrid(
    np.linspace(-camera_height / 2, camera_height / 2, num=camera_height),
    np.linspace(-camera_width / 2, camera_width / 2, num=camera_width)
)
reference_w0 = 50


def angle(x, y):
    return np.arctan2(x, y)


def LaguerreGauss(p, m, x=X, y=Y, w0=reference_w0):
    laguerreP = scipy.special.genlaguerre(p, np.abs(m))
    R2 = x**2 + y**2
    lg = 0 * 1j + np.sqrt((2**(np.abs(m) + 1) * scipy.special.gamma(1 + p)) /
                          (np.pi * w0**2 * scipy.special.gamma(1 + p + np.abs(m))))
    lg *= (np.sqrt(R2) / w0)**np.abs(m)
    lg *= np.exp(-R2 / (w0**2)) * laguerreP((2 * R2) / w0**2)
    # additional phase depending on `m` dof
    lg *= np.exp(- 1j * m * angle(x, y))
    return lg


def _bloch_sphere_vector(theta, phi):
    """Amplitudes of Bloch sphere vector in usual notation."""
    return np.asarray([
        np.cos(theta / 2),
        np.sin(theta / 2) * np.exp(1j * phi)
    ])


def _su2_basis_states(which):
    """Return commonly used basis states from a name."""
    if not isinstance(which, (list, tuple)):
        which = [which]
    answers = []
    for a_which in which:
        if a_which == 'H' or a_which == '0':
            answers.append(np.array([1, 0]))
        elif a_which == 'V' or a_which == '1':
            answers.append(np.array([0, 1]))
        elif a_which == 'L':
            answers.append([1, -1j] / np.sqrt(2))
        elif a_which == 'R':
            answers.append([1, 1j] / np.sqrt(2))
        elif a_which == 'D' or a_which == '+':
            answers.append([1, 1] / np.sqrt(2))
        elif a_which == 'A' or a_which == '-':
            answers.append([1, -1] / np.sqrt(2))
    return answers


def hyperentangled_qubit_projection(qubit_amps, qudits_amps, projectors):
    """Project the qubit dof in the hyperentangled state.
    
    Attributes
    ----------
    qubit_amps : pair of complex numbers
        The complex coefficients of the qubit state.
    qudits_amps : pair of lists
        The qudit states associated with the qubit states. The full state is
        assumed to be
            qubit_amps[0] |0, qudits_amps[0]> +
            qubit_amps[1] |1, qudits_amps[1]>.
    projectors : list of pairs of complex numbers
        Each element should be a pair representing the qubit state that we
        want the overall state to be projected on.

    """
    probs = []
    for projector_amps in projectors:
        probs.append(abs2(
            np.conjugate(projector_amps[0]) * qubit_amps[0] * qudits_amps[0] +
            np.conjugate(projector_amps[1]) * qubit_amps[1] * qudits_amps[1]
        ))
    return probs


def probabilities_to_stokes_parameters(list_of_probabilities):
    """Compute stokes parameters from a set of six probability vectors."""
    p0, p1, p_plus, p_minus, p_L, p_R = list_of_probabilities
    return np.asarray([p0 - p1, p_plus - p_minus, p_R - p_L])


def vector_vortex_stokes_pars(X, Y, p, m_pair, w0, polarization_state):
    """Return the Stokes probability vectors for a given vector vortex beam.
    
    Attributes
    ----------
    m_pair : tuple of ints
        A vector vortex beam is a superposition c0 |m1> + c1 |m2>, and `m_pair`
        contains m1 and m2.
    polarization_state : pair of floats
        Coefficients of the polarization state that is entangled with the OAM.
        We assume the polarization state is pure, and the parameters are the
        complex coefficient in the computational basis.
        In other words, these are the values of c0 and c1 above.
    
    Returns
    -------
    The set of three Stokes probability vectors associated with the state.
    This is a numpy array of shape (3, N, M), where N, M = X.shape = Y.shape
    """
    if len(m_pair) != 2:
        raise ValueError('There must be two elements in `m_pair`.')

    c0, c1 = polarization_state
    amps_m1 = c0 * LaguerreGauss(X, Y, p=p, m=m_pair[0], w0=w0)
    amps_m2 = c1 * LaguerreGauss(X, Y, p=p, m=m_pair[1], w0=w0)
    average_Z = abs2(amps_m1) - abs2(amps_m2)
    twice_c0star_times_c1 = 2 * np.conj(amps_m1) * amps_m2
    average_X = np.real(twice_c0star_times_c1)
    average_Y = np.imag(twice_c0star_times_c1)
    return np.array([average_Z, average_X, average_Y])


def rotation_matrix(theta):
    """Compute 2D rotation matrix given an angle in degrees."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def rotated_waveplate_matrix(theta, waveplate_matrix):
    """Compute matrix representing a rotated waveplate.

    Angles are to be given in degrees.
    """
    return np.dot(rotation_matrix(theta),
                  np.dot(waveplate_matrix, rotation_matrix(-theta)))


def rotated_HWP_matrix(angle):
    """Compute matrix representing a rotated half-waveplate.

    Angles are to be given in degrees.
    """
    HWP = np.array([[1, 0], [0, -1]])
    return rotated_waveplate_matrix(angle, HWP)


def rotated_QWP_matrix(angle):
    """Compute matrix representing a rotated quarter-waveplate.

    Angles are to be given in degrees.
    """
    QWP = np.array([[1, 0], [0, 1j]])
    return rotated_waveplate_matrix(angle, QWP)


def polarization_projection_matrix_from_waveplates(alpha_HWP, alpha_QWP):
    """Compute polarization projectors with rotated waveplates matrices.
    """
    return np.dot(rotated_HWP_matrix(alpha_HWP), rotated_QWP_matrix(alpha_QWP))


# def accuracies_from_predictions(true_labels, predicted_labels, labels_names):
#     """Build dictionary of accuracies from true and predicted labels.

#     Attributes
#     ----------
#     true_labels : array of ints
#         Array of ints of shape (num_samples,). Each element contains an int
#         that represent one of the labels in `labels_names`. It represents the
#         true labels associated with the elements of a dataset.
#     predicted_labels : array of ints
#         Same as true_labels, but representing the labels that were predicted
#         via some classification algorithm on the same dataset.
#     labels_names : list of strings
#         Each elements contains the name of a label. The ints of true_labels
#         are assumed to represent elements from this list.
#     """
#     accuracies_per_label = collections.OrderedDict()
#     for label_idx, label_name in enumerate(labels_names):
#         # initialize list of number of predicted labels per each true label
#         accuracies = [0] * len(labels_names)
#         # extract predictions for the specific label
#         predictions = predicted_labels[true_labels == label_idx]
#         print(label_idx, label_name, predictions)
#         # accuracies = list(collections.Counter(predictions).values())
#         for outlabel_idx, count in collections.Counter(predictions).items():
#             accuracies[outlabel_idx] = count
#         accuracies_per_label[label_name] = accuracies
#     return accuracies_per_label


class ReduceAndClassify:
    def __init__(self, labeled_dataset=None):
        """Initialize class instance.

        Attributes
        ----------
        labeled_dataset : dict
            Each element of the dictionary should correspond to a dataset
            associated with a different labels. The keys are taken to be the
            names of the different labels.
        """
        self.labels_names = []
        self.labels = None
        self.dataset = None
        self.reduced_dataset = None
        self.pca = None
        self.svc = None

        if labeled_dataset is not None:
            self.add_dataset(labeled_dataset)
    
    def add_dataset(self, labeled_dataset):
        labeled_dataset = collections.OrderedDict(labeled_dataset)
        new_labels_names = list(labeled_dataset.keys())
        # if there are new labels, add them to the list
        for new_label_name in new_labels_names:
            if new_label_name not in self.labels_names:
                self.labels_names += [new_label_name]
        # separate labels and data and append as appropriate
        for label, data in labeled_dataset.items():
            # append new label information
            new_labels = [self.labels_names.index(label)] * data.shape[0]
            if self.labels is None:
                self.labels = new_labels
            else:
                self.labels += new_labels
            # append new data to dataset
            if self.dataset is None:
                self.dataset = data
            else:
                self.dataset = np.concatenate((self.dataset, data), axis=0)

    def apply_PCA(self, n_components, **kwargs):
        self.pca = sklearn.decomposition.PCA(n_components, **kwargs)
        self.reduced_dataset = self.pca.fit_transform(self.dataset)
    
    def fit_SVM(self, num_dimensions=None, **kwargs):
        if self.reduced_dataset is None or self.pca is None:
            raise ValueError('Apply PCA before doing this.')
        # use previously trained PCA to reduce dimensions
        if num_dimensions is None:
            num_dimensions = self.reduced_dataset.shape[1]

        # if the number of requested dimensions is smaller than the one used
        # previously for the dimensionality reduction, take only the relevant
        # first dimensions
        if num_dimensions < self.reduced_dataset.shape[1]:
            reduced_dataset = self.reduced_dataset[:, :num_dimensions]
        else:
            reduced_dataset = self.reduced_dataset

        # fucking classify already
        clf = sklearn.svm.SVC(**kwargs)
        clf.fit(reduced_dataset, self.labels)
        self.svc = clf
    
    def reduce_and_classify(self, data):
        """Apply PCA SVC to new data.
        
        Attributes
        ----------
        data : numpy array
            The shape must be (num_samples, dim_classifier), where dim_classifier
            is the dimension of the vectors on which the SVC was trained.
        """
        if self.svc is None:
            raise ValueError('The classifier must be trained before this.')
        num_dimensions = self.svc.support_vectors_.shape[1]
        reduced_data = self.pca.transform(data)[:, :num_dimensions]
        return self.svc.predict(reduced_data)

    def cut_feature_space_components(self, data, num_dimensions=None):
        if self.pca is None:
            raise ValueError('The classifier must be trained before this.')
        reduced_data = self.pca.transform(data)
        if num_dimensions is not None:
            reduced_data[:, num_dimensions:] = 0
        return self.pca.inverse_transform(reduced_data)


class OAMDataset(ReduceAndClassify):
    def __init__(self, X, Y, w0):
        self.X = X
        self.Y = Y
        self.w0 = w0
        super().__init__()

    def _generate_data_one_type(self, pm_list, num_samples, noise_level):
        """Generate OAM states superposition of given p and m parameters."""
        data = np.zeros(shape=(num_samples, len(self.X) * len(self.Y)))
        for idx in range(num_samples):
            amps = np.zeros(shape=len(self.X) * len(self.Y), dtype=np.complex)
            for p, m in pm_list:
                amps += LaguerreGauss(
                    self.X, self.Y, p=p, m=m, w0=self.w0).flatten()
            data[idx] = utils.add_noise_to_array(abs2(amps), noise_level=noise_level)
        return data

    def generate_data(self, parameters, num_samples=50, noise_level=0.1,
                      monitor=False, polarization_state='random phases'):
        """Generate vectors corresponding to OAM states.
        
        Attributes
        ----------
        parameters : list of tuples
            Each element should be a list of pairs [(p1, m1), (p2, m2), ...].
        """
        dataset = collections.OrderedDict()
        labels_names = []

        iterator = range(len(parameters))
        # monitoring with progressbar only work with jupyter notebooks, while
        # for jupyterlab only the tqdm one works well, hence the choice here
        if monitor and isinstance(monitor, str) and monitor == 'progressbar':
            bar = progressbar.ProgressBar()
            iterator = bar(iterator)
        elif monitor:
            iterator = tqdm(iterator)
        # loop over the parameters given to generate the data
        for par_idx in iterator:
            parameter_list = parameters[par_idx]
            dataset[tuple(parameter_list)] = self._generate_data_one_type(
                pm_list=parameter_list,
                num_samples=num_samples, noise_level=noise_level
            )
            labels_names.append(str(tuple(parameter_list)))

        self.add_dataset(dataset)


class VVBDataset(ReduceAndClassify):
    def __init__(self, X, Y, w0):
        self.X = X
        self.Y = Y
        self.w0 = w0
        super().__init__()

    def _generate_data_one_type(self, p, m_pair, num_samples, noise_level,
                                polarization_state):
        data = np.zeros(shape=(num_samples, len(self.X) * len(self.Y) * 3))
        for idx in range(num_samples):
            if isinstance(polarization_state, str):
                if polarization_state == 'random phases':
                    phi = 1j * 2 * np.pi * np.random.rand(1)
                    c0 = 1 / np.sqrt(2)
                    c1 = c0
                elif polarization_state == 'sequential phases':
                    phi = 1j * 2 * np.pi * idx / num_samples
                    c0 = 1 / np.sqrt(2)
                    c1 = c0
                elif polarization_state == 'uniform random':
                    phi = 1j * 2 * np.pi * np.random.rand(1)
                    theta = np.pi * np.random.rand(1)
                    c0 = np.cos(theta)
                    c1 = np.sin(theta)
                else:
                    raise ValueError('Unrecognised option')
                pol_state = [c0, c1 * np.exp(phi)]
            else:
                pol_state = polarization_state

            probs = vector_vortex_stokes_pars(
                X=self.X, Y=self.Y, p=p,
                m_pair=m_pair, w0=self.w0,
                polarization_state=pol_state
            ).flatten()
            data[idx] = utils.add_noise_to_array(probs, noise_level=noise_level)
        return data

    def generate_data(self, parameters, num_samples=50, noise_level=0.1,
                      monitor=False, polarization_state='random phases'):
        """Generate vectors corresponding to Stoke parameters.
        
        Attributes
        ----------
        parameters ; list of tuples
            Each element should be a pair (p, (m1, m2)).
        """
        dataset = collections.OrderedDict()
        labels_names = []

        iterator = range(len(parameters))
        if monitor:
            iterator = tqdm(iterator)
        # loop over the parameters given to generate the data
        for par_idx in iterator:
            parameter = parameters[par_idx]
            dataset[parameter] = self._generate_data_one_type(
                p=parameter[0], m_pair=parameter[1],
                num_samples=num_samples, noise_level=noise_level,
                polarization_state=polarization_state
            )
            labels_names.append(str(parameter))

        self.add_dataset(dataset)



