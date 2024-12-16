## functions for mTE analysis
#
# written by S-C. Baek
# update: 16.12.2024
#
"""
Collection of functions to run multivariate transfer entropy (mTE) analysis.
The code here were mainly created by tranlating the original MATLAB code in the following Github repository:
https://github.com/ide2704/Kernel_Renyi_Transfer_Entropy/tree/master.
The translated code were further adapted and customized for our purposes.
"""
import copy

import numpy as np

from scipy.spatial.distance import pdist, cdist

import contextlib
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

from utils import _multiple_matrix_multiply

def _check_symmetric(mtx, rtol=1e-05, atol=1e-08):
    """check whether a given matrix is symmetric"""
    return np.allclose(mtx, mtx.T, rtol=rtol, atol=atol)


def _kernel_entropy(data, alpha=2.0):
    """
    This function estimate Renyi'a alpha entropy based on a kernel matrix.
    This code is a translation of the origianl MATLAB code here:
    https://github.com/ide2704/Kernel_Renyi_Transfer_Entropy/tree/master.
    """
    data = np.array(data)
    assert data.ndim == 2
    assert _check_symmetric(data)

    # compute kernel-based Reyni entropy
    data = data / np.trace(data)  # normalize the input matrix
    L, _ = np.linalg.eigh(data)
    H = np.real((1 / (1 - alpha)) * np.log2(np.sum(L ** alpha)))
    return H


def _rbf_kernel_matrix(data, sigma=None):
    """
    This function computes a gram matrix of data by applying radial basis function kernel (RBF; gaussian).
    Other kernel functions are currently not supported.

    This function is created by translating and adapting the original MATLAB code here:
    https://github.com/ide2704/Kernel_Renyi_Transfer_Entropy/tree/master.
    """
    data = np.array(data)
    if data.ndim == 1:  # make data into a n_times-by-n_dists form
        data = np.atleast_2d(data).T
    elif data.ndim > 2:
        raise ValueError(" Data larger than 3D is not supported ... ")

    # estimate the width of a gaussian kernel from data
    if sigma is None:
        sigma = np.median(pdist(data))

    # compute RBF kernel
    K = np.exp(-1 * cdist(data, data) ** 2 / (2 * sigma ** 2))
    assert _check_symmetric(K)  # check if an output matrix is symmetric

    return K


def kR_te(x, y, x_prev, alpha=2.0):
    """
    This function computes multivariate transfer entropy by using kernel-based Renyi's alpha entropy.

    This function is created by translating and adapting the original MATLAB code here:
    https://github.com/ide2704/Kernel_Renyi_Transfer_Entropy/tree/master.

    Input
    ----------
    x : ndarray, shape (n_times, n_dists)
        Time series of neural RDMs from the target ROI at t.
    y : ndarray, shape (n_times, n_dists)
        Time series of neural RDMs from the source ROI at t-l (l: time lag).
    x_prev : ndarray, shape (n_times, n_dists)
        Time series of neural RDMs from the target ROI at t-l (l: time lag).
    alpha : float
        A hyperparameter for Renyi's alpha entropy.
        Defaults to 2.0 for neutral weighting between rare and common events.

    Returns
    -------
    te : float
        mTE value computed by matrix-based Renyi's alpha entropy at a give time lag from a given directed connection.
    """
    x = np.array(x)  # target
    assert x.ndim < 3
    y = np.array(y)  # source
    assert y.ndim < 3
    x_prev = np.array(x_prev)  # target previous
    assert x_prev.ndim < 3

    # check if the lengths of input match
    Lx, Ly = len(x), len(y)  # n_samples
    assert Lx == Ly
    assert Lx == len(x_prev)

    # compute kernel matrices
    K_x = _rbf_kernel_matrix(x)
    K_y = _rbf_kernel_matrix(y)
    K_x_prev = _rbf_kernel_matrix(x_prev)

    # joint and marginal kernel-based Renyi entropy estimation
    K_yz = _multiple_matrix_multiply([K_y, K_x_prev])
    H_yz = _kernel_entropy(K_yz, alpha=alpha)

    K_xyz = _multiple_matrix_multiply([K_x, K_y, K_x_prev])
    H_xyz = _kernel_entropy(K_xyz, alpha=alpha)

    K_xz = _multiple_matrix_multiply([K_x, K_x_prev])
    H_xz = _kernel_entropy(K_xz, alpha=alpha)

    H_z = _kernel_entropy(K_x_prev, alpha=alpha)

    # compute transfer entropy
    te = H_yz - H_xyz + H_xz - H_z

    return te


def mte_single(target, source, lag, max_lag):
    """
    This function computes transfer entropy given two multivariate time series at a given time lag.

    Input
    ----------
    target : ndarray, shape (n_times, n_dists)
        Full time series of neural RDMs from the target ROI (sample points from stimulus onset).
    source : ndarray, shape (n_times, n_dists)
        Full time series of neural RDMs from the source ROI (sample points from stimulus onset).
    lag : int
        A parameter for a time lag, based on which source and target neural RDMs are delayed.
    max_lag : int
        A maximum time lag in the analysis. To validate 'lag'.

    Returns
    -------
    te : float
        mTE value computed from the source to target at a given time lag.
    """
    # make sure that lag parameters are int
    lag = int(lag)
    max_lag = int(max_lag)

    # n_times
    n_times = target.shape[0]

    # copy data to model previous time series of target and source
    source_prev = copy.deepcopy(source)
    target_prev = copy.deepcopy(target)

    # check the validity of time lag
    assert int(n_times // 2) >= max_lag

    # introduce a time lag to source and target time series
    target_lagged = target[lag:lag - max_lag or None]
    source_prev_lagged = source_prev[:-max_lag]
    target_prev_lagged = target_prev[:-max_lag]

    # compute mTE
    te = kR_te(target_lagged, source_prev_lagged, target_prev_lagged)

    return te


def compute_mte(target, source, max_lag, lag_step=1, n_jobs=1, verbose=False):
    """
    This function computes transfer entropy given two uni-/multi-variate time series at variable time lags.

    Input
    ----------
    target : ndarray, shape (n_times, n_dists)
        Full time series of neural RDMs from the target ROI (sample points from stimulus onset).
    source : ndarray, shape (n_times, n_dists)
        Full time series of neural RDMs from the source ROI (sample points from stimulus onset).
    max_lag : int
        A maximum time lag that is considered in the analysis.
    lag_step : int
        For specifying how much a time lag increases at a time.
        Defaults to 1
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to use all available cores.
        Defaults to 1.
    verbose : bool
        If specified, a progress bar appears at the prompt.

    Returns
    -------
    mte_vals : ndarray, shape ([n_lags,])
        An array of the mTE value at each time lag.
    """
    # check data structure
    target = np.array(target)
    assert target.ndim == 2
    source = np.array(source)
    assert source.ndim == 2
    assert target.shape[0] == source.shape[0]

    # list of lags
    max_lag = int(max_lag)
    lags = np.arange(1, max_lag+1, lag_step) # spanning from 1 to max_lag

    # whether to display the progress
    if verbose:
        @contextlib.contextmanager
        def tqdm_joblib(tqdm_object):
            """
            Context manager to patch joblib to report into tqdm progress bar given as argument.
            Copied from here:
            https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib.
            """

            class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
                def __call__(self, *args, **kwargs):
                    tqdm_object.update(n=self.batch_size)
                    return super().__call__(*args, **kwargs)

            old_batch_callback = joblib.parallel.BatchCompletionCallBack
            joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
            try:
                yield tqdm_object
            finally:
                joblib.parallel.BatchCompletionCallBack = old_batch_callback
                tqdm_object.close()

        with tqdm_joblib(tqdm(desc=" Lags ", total=len(lags))) as pbar:
            # Call RSA multiple times in parallel for each searchlight patch
            data = Parallel(n_jobs)( delayed(mte_single)(target, source, l, max_lag) for l in lags )
    else: # without progress bar
        data = Parallel(n_jobs)( delayed(mte_single)(target, source, l, max_lag) for l in lags )

    return np.array(data)