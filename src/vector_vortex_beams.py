import math
import os
import sys

import numpy as np
import pandas as pd
import scipy
import scipy.special

import keras
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow
from PIL import Image

from utils import abs2



def angle(x, y):
    return np.arctan2(x, y)


def LaguerreGauss(x, y, p, m, w0):
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
    
    Returns
    -------
    The set of three Stokes probability vectors associated with the state.
    """
    if len(m_pair) != 2:
        raise ValueError('There must be two elements in `m_pair`.')
    amps_first_m = LaguerreGauss(X, Y, p, m_pair[0], w0)
    amps_second_m = LaguerreGauss(X, Y, p, m_pair[1], w0)

    states_to_project_upon = _su2_basis_states(['0', '1', '+', '-', 'R', 'L'])
    probabilities_for_basis_states = hyperentangled_qubit_projection(
        qubit_amps=polarization_state,
        qudits_amps=[amps_first_m, amps_second_m],
        projectors=states_to_project_upon
    )
    return probabilities_to_stokes_parameters(probabilities_for_basis_states)


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

