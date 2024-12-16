## utility functions
#
# written by S-C. Baek
# update: 16.12.2024
#
# encoding: utf-8
'''
collection of utility functions.
'''
import os
import pickle

import numpy as np

from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix, coo_matrix

from mne.stats.multi_comp import fdr_correction


def get_main_dir():
    '''get the main directory of the repository'''
    # path settings
    PWD = os.getcwd()
    if PWD.split('/')[-1] == 'python':
        MAINDIR = os.path.abspath('../..')
    else:
        MAINDIR = os.path.abspath('..')
    files = os.listdir(MAINDIR)

    # check if MAINDIR is actually the main directory
    assert 'code' in files
    assert 'data' in files
    assert 'figs' in files
    assert 'stim' in files

    return MAINDIR

def _multiple_matrix_multiply(data):
    """This function computes Hadamard product of a given list of matrices, apply numpy.multiply recursively"""
    assert isinstance(data, list)
    assert data[0].ndim == 2
    if len(data) > 1:
        ref = data[0]
        for d in data[1:]:
            assert ref.shape == d.shape
        del ref, d

    # compute Hamadard product recursively
    for i in range(len(data)):
        if i == 0:
            hamadard_product = data[0]
        else:
            hamadard_product = np.multiply(hamadard_product, data[i])

    return hamadard_product


def pairwise_contrast_sparse(index_vector) -> csr_matrix:
    """
    This function is taken from 'rsatoolbox.util.matrix' module.
    (https://rsatoolbox.readthedocs.io/en/stable/rsatoolbox.util.matrix.html)

    It contrasts matrix with one row per unique pairwise contrast.

    Args:
        index_vector (numpy.ndarray): n_row vector to code
            discrete values (one dimensional)

    Returns:
        scipy.sparse.csr_matrix: indicator_matrix:
            n_values * (n_values-1)/2 x n_row contrast matrix
    """
    c_unique = np.unique(index_vector)
    n_unique = c_unique.size
    rows = np.size(index_vector)
    cols = int(n_unique * (n_unique - 1) / 2)
    # Now make an indicator_matrix with a pair of conditions per row
    n_repeats = np.zeros(n_unique, dtype=int)
    select = [None] * n_unique
    for i in range(n_unique):
        sel = (index_vector == c_unique[i])
        n_repeats[i] = np.sum(sel)
        select[i] = list(np.where(index_vector == c_unique[i])[0])
    n_row = 0
    dat = []
    idx_i = []
    idx_j = []
    for i in range(n_unique):
        for j in np.arange(i + 1, n_unique):
            dat += [1/n_repeats[i]] * n_repeats[i]
            idx_i += [n_row] * n_repeats[i]
            idx_j += select[i]
            dat += [-1/n_repeats[j]] * n_repeats[j]
            idx_i += [n_row] * n_repeats[j]
            idx_j += select[j]
            n_row = n_row + 1
    indicator_matrix = coo_matrix((dat, (idx_i, idx_j)),
                                  shape=(cols, rows))
    return indicator_matrix.asformat("csr")


def get_v(n_conds, sigma_k):
    """
    This function is taken from 'rsatoolbox.util.matrix' module.
    (https://rsatoolbox.readthedocs.io/en/stable/rsatoolbox.util.matrix.html)

    It gets the rdm covariance from noise covariance matrix across conditions.
    """
    # calculate pairwise contrast matrix between conditions
    c_mat = pairwise_contrast_sparse(np.arange(n_conds))
    if sigma_k is None:
        xi = c_mat @ c_mat.transpose()
    else:
        sigma_k = csr_matrix(sigma_k)
        xi = c_mat @ sigma_k @ c_mat.transpose()
    # calculate V
    v = xi.multiply(xi).tocsc()
    v = np.array(v.todense())

    return v


def fisher_z_transform(X, sqrt=True):
    X = np.array(X)
    if sqrt:
        X[X < 0] = 0.0 # zeroing negative values
        X = np.sqrt(X)
    return np.arctanh(X)


def whitening_rdms(data, cov):
    '''
    This function whitens neural RDMs.
    data : currently supports the shape: (n_rois), n_times, n_dists
    cov : currently only supports the following - sigma_k for each rois (n_conds X n_conds)

    Input
    ----------
    data : ndarray, shape ([n_times,] [n_dists,]) | ([n_rois,] [n_times,] [n_dists,])
        An array of neural RDMs.
    cov : ndarray, shape (n_cond, n_cond) | list of ndarray, whose elements correspond to n_rois.
        A weight matrix correspond to data.

    Returns
    ----------
    data_white : ndarray, shape ([n_times,] [n_dists,]) | ([n_rois,] [n_times,] [n_dists,])
        An array of neural RDMs, whitened by weight matrices derived from the corresponding 'cov.'
    '''
    # when cov is not list
    if not isinstance(cov, list):
        cov = [cov]

    # check the data structure of cov
    for c in cov:
        assert c.ndim == 2 and c.shape[0] == c.shape[-1]

    # depending on the shape of data
    if data.ndim == 3 and len(cov) > 1: # when data is n_rois, n_times, n_dists
        assert len(data) == len(cov)
    elif data.ndim == 3 and len(cov) == 1: # apply the same cov for all rois
        cov = cov * data.shape[0]
        assert len(data) == len(cov)
    elif data.ndim == 2:
        data = data[np.newaxis,:,:]
        assert len(data) == len(cov)
    n_rois = len(data)

    # the last dimension of data should correspond to the distances between conditions
    n_dists = data.shape[-1]
    n_conds = squareform(np.ones(n_dists)).shape[0]  # number of conditions
    assert n_conds == cov[0].shape[0]

    # loop over rois
    data_white = np.zeros(data.shape)
    for roii in range(n_rois):

        # compute V matrix
        V = get_v(n_conds, cov[roii])

        # compute whitening matrix
        L, K = np.linalg.eigh(V)
        inv_l = 1 / np.sqrt(np.abs(L))
        W = K @ np.diag(inv_l) @ K.T  # V^-1/2 for whitening

        # apply whitening matrix
        data_white[roii, :, :] = np.einsum('ij,jk->ik', data[roii], W)
        del W

    if data_white.shape[0] == 1:
        data_white = data_white[0]

    return data_white


def sort_rois_subset(element, between_conditions=False, p_threshold=0.05):
    """
    This function sorts out a subset out of 10 ROIs based on general representations
    with respect to phonemes or prosody.

    Input
    ----------
    element : str
        A linguistic element to be analysis. it should be either 'phoneme' or 'prosody'.

    Returns
    ----------
    rois_subset_indices : ndarray, shape (n_rois_subset,)
        Indices of ROIs sorted out from statistical results (after applying FDR correction).
    """
    # check the input
    if element not in ['phoneme', 'prosody']:
        raise ValueError("Invalid element, must be one of: 'phoneme' or 'prosody' ")

    # path settings
    MAINDIR = get_main_dir()
    DATADIR = MAINDIR + os.sep + 'data/'

    # import ROI-wise cluster statistics
    fname = DATADIR + 'group' + os.sep + 'meg' + os.sep + 'rsa_rois_' + element + '_gen'
    fname += '_att_vs_ign' if between_conditions else '_vs_baseline'
    fname += '_p' + str(p_threshold)[2:] + '.pickle'
    with open(fname, 'rb') as f:
        cluster_stats_rois = pickle.load(f)
    n_rois = len(cluster_stats_rois)

    # FDR correction based on the minimum p-value of each ROI
    pvals = list()
    for cluster_stats in cluster_stats_rois:
        if len(cluster_stats[1]) > 0:
            pvals.append(np.min(cluster_stats[1]))
        else:
            pvals.append(0.99)
    reject, _ = fdr_correction(pvals)  # apply FDR correction

    return np.arange(n_rois)[reject]