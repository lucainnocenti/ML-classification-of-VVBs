import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.special
import seaborn as sns
from PIL import Image

import keras
import theano


def angle(x, y):
    return np.arctan2(x, y)


def LaguerreGauss(x, y, p, m, w0=1.):
    laguerreP = scipy.special.genlaguerre(p, np.abs(m))
    R2 = x**2 + y**2
    lg = 0 * 1j + np.sqrt((2**(np.abs(m) + 1) * scipy.special.gamma(1 + p)) /
                          (np.pi * w0**2 * scipy.special.gamma(1 + p + np.abs(m))))
    lg *= (np.sqrt(R2) / w0)**np.abs(m)
    lg *= np.exp(-R2 / (w0**2)) * laguerreP((2 * R2) / w0**2)
    # additional phase depending on `m` dof
    lg *= np.exp(- 1j * m * angle(x, y))
    return lg


def plot_intensity(amplitudes, imshow_opts={}, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    intensities = np.abs(amplitudes)**2
    ax.imshow(intensities, interpolation='nearest', cmap='magma',
              origin='lower', **imshow_opts)
    # ax.axis('off')


def rotation_matrix(theta):
    """Compute 2D rotation matrix given an angle in degrees (NOT radians)."""
    return np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                     [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])


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


def polarization_basis_states(which):
    """Return commonly used basis states from a name."""
    if which == 'H':
        return np.array([1, 0])
    elif which == 'V':
        return np.array([0, 1])
    elif which == 'L':
        return [1, -1j] / np.sqrt(2)
    elif which == 'R':
        return [1, 1j] / np.sqrt(2)
    elif which == 'D':
        return [1, 1] / np.sqrt(2)
    elif which == 'A':
        return [1, -1] / np.sqrt(2)
