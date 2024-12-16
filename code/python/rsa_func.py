## functions for RSA.
#
# written by S-C. Baek
# update: 16.12.2024
#
"""
Collection of functions to implement representational similarity analysis (RSA).
The code below is customized based on 'mne-rsa 0.8dev (https://users.aalto.fi/~vanvlm1/mne-rsa/#development).'
"""
import numpy as np

from scipy.spatial import distance
from sklearn.linear_model import LinearRegression

import contextlib
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm


def _ensure_condensed(rdm, var_name):
    """
    This function is taken from mne-rsa 0.8dev.
    It converts a single RDM to a condensed form if needed.
    """
    if type(rdm) is list:
        return [_ensure_condensed(d, var_name) for d in rdm]

    if not isinstance(rdm, np.ndarray):
        raise TypeError('A single RDM should be a NumPy array. '
                        'Multiple RDMs should be a list of NumPy arrays.')

    if rdm.ndim == 2:
        if rdm.shape[0] != rdm.shape[1]:
            raise ValueError(f'Invalid dimensions for "{var_name}" '
                             '({rdm.shape}). The RDM should either be a '
                             'square matrix, or a one dimensional array when '
                             'in condensed form.')
        rdm = distance.squareform(rdm)
    elif rdm.ndim != 1:
        raise ValueError(f'Invalid dimensions for "{var_name}" ({rdm.shape}). '
                         'The RDM should either be a square matrix, or a one '
                         'dimensional array when in condensed form.')
    return rdm


def _nested_comp_uvar_nnls(neural_rdm, rdm_model, weighting=None):
    """
    Compute unique variance explained by each model using nested model comparison.
    Linear regression is performed based on non-negative least squares (NNLS).
    Full model is based on all model RDMs available.
    Reduced model is based on all model RDMs except for on being tested.
    More than one model RDMs are required.
    If a weight matrix is specified, whitening distance estimates is performed.
    """
    if len(rdm_model) == 1:
        raise ValueError('Need more than one model RDM to use '
                         'nested-comp-uvar-nnls as metric.')
    n_models = len(rdm_model)

    # number of distances
    n_dists = len(neural_rdm)

    # if a weight matrix is specified
    if weighting is not None:
        # to make sure a weight matrix is ndarray
        V = np.array(weighting)

        # compute whitening matrix
        L, K = np.linalg.eigh(V)
        inv_l = 1 / np.sqrt(np.abs(L))
        W = K @ np.diag(inv_l) @ K.T  # V^-1/2 for whitening
    else:
        W = np.eye(n_dists)

    # ---------- full model ---------- #
    X = np.atleast_2d(np.array([np.ones(n_dists)] + rdm_model)).T  # design metrix including intercept
    Y = np.atleast_2d(np.array(neural_rdm)).T  # dependent variable

    # weighting X and Y with decorrelation matrix
    wX = np.array(W @ X)
    wY = np.array(W @ Y)

    # main functionality
    reg_nnls_full = LinearRegression(fit_intercept=False, positive=True).fit(wX, wY)
    rsq_full = reg_nnls_full.score(wX, wY) # r_squared of full model

    # init output array
    rsq_reduceds = np.empty( (n_models, ) )

    # loop over rdm_model
    for i in range(n_models):

        # Reduced model
        reduced_model = rdm_model[:i] + rdm_model[i + 1:]
        X_reduced = np.atleast_2d(np.array([np.ones(n_dists)]+reduced_model)).T # design metrix including intercept
        wX_reduced = np.array(W @ X_reduced)  # weighting X_reduced with decorrelation matrix
        reg_nnls_reduced = LinearRegression(fit_intercept=False, positive=True).fit(wX_reduced, wY)
        rsq_reduceds[i] = reg_nnls_reduced.score(wX_reduced, wY) # r_squared of reduced model

    # specify the output
    rsq_reduceds[rsq_reduceds < 0] = 0
    out = rsq_full - rsq_reduceds # unique variance explained by each parameter

    return out


def _linear_regression_rsq_nnls(neural_rdm, rdm_model, weighting=None):
    """
    Compute r-squared by linear regression based on NNLS (with an effect of intercept excluded).
    A linear regression model takes a single neural RDM as a dependent variable,
    and one or multiple model RDMs as independent variable.
    If a weight matrix is specified, whitening distance estimates is performed.
    """
    # number of distances
    n_dists = len(neural_rdm)

    # number of models
    n_models = len(rdm_model)

    # if a weight matrix is specified
    if weighting is not None:
        # to make sure a weight matrix is ndarray
        V = np.array(weighting)

        # compute whitening matrix
        L, K = np.linalg.eigh(V)
        inv_l = 1 / np.sqrt(np.abs(L))
        W = K @ np.diag(inv_l) @ K.T  # V^-1/2 for whitening
    else:
        W = np.eye(n_dists)

    # ---------- full model ---------- #
    X = np.atleast_2d(np.array([np.ones(n_dists)] + rdm_model)).T  # design metrix including intercept
    Y = np.atleast_2d(np.array(neural_rdm)).T  # dependent variable

    # weighting X and Y with decorrelation matrix
    wX = np.array(W @ X)
    wY = np.array(W @ Y)

    # main functionality
    reg_nnls = LinearRegression(fit_intercept=False, positive=True).fit(wX, wY)
    rsq = reg_nnls.score(wX, wY) # r_squared of full model

    # ---------- null model ---------- #
    X0 = np.atleast_2d(np.array([np.ones(n_dists)])).T  # intercept only model

    # weighting X0 with decorrelation matrix
    wX0 = np.array(W @ X0)

    # main functionality
    reg_nnls = LinearRegression(fit_intercept=False, positive=True).fit(wX0, wY)
    rsq0 = reg_nnls.score(wX0, wY) # r_squared of null model

    # subtract rsq0 from rsq if rsq0 is larger then 0
    if rsq0 > 0:
        rsq = rsq - rsq0

    return rsq * np.ones( (n_models,) ) # to meet the shape criteria


def _rsa_single_rdm(neural_rdm, rdm_model, metric, weighting=None):
    """Compute RSA between a single neural RDM and model RDMs."""
    if metric == 'regression-rsq-nnls':
        rdm_model = [rdm for rdm in rdm_model]
        rsa_vals = _linear_regression_rsq_nnls(neural_rdm, rdm_model, weighting)
    elif metric == 'nested-comp-uvar-nnls': # added by S-C. Baek
        rdm_model = [rdm for rdm in rdm_model]
        rsa_vals = _nested_comp_uvar_nnls(neural_rdm, rdm_model, weighting)
    else:
        raise ValueError("Invalid RSA metric, must be one of: 'nested-comp-uvar-nnls' or 'regression-rsq-nnls' ")
    return rsa_vals


def rsa_array(rdm_array, rdm_model, rsa_metric='regression-rsq-nnls', weights=None, n_jobs=1, verbose=True):
    """
    This function is adapted from rsa_arry in mne-rsa 0.8dev.
    It Performs RSA on an array of data.

    Input
    ----------
    rdm_array : ndarray, shape (n_vertices, n_times, n_features)
        An array of precomputed neural RDMs.
    rdm_model : ndarray, shape (n_cond, n_cond) | (n_cond * (n_cond - 1) // 2,) | list of ndarray
        The model RDM(s). Both square (`scipy.spatial.distance.squareform`) and condensed forms are possible.
        A condensed form corresponds to the upper triangualr entries of a square form.
        For using multiple model RDMs, they should be provided as list.
    rsa_metric : str
        The RSA metric to use to compare neural and model RDMs.
        Valid options are:
        * 'regression-rsq-nnls' for r-squared by all models using linear regression based on NNLS.
        * 'nested-comp-uvar-nnls' for unique variance explained by each model using nested linear modeling based on NNLS.
        Defaults to 'regression-rsq-nnls'.
    weights : ndarry, shape (n_cond, n_cond) | list of ndarray
        A weight matrix or list of weight matrices to whiten distance estimates.
        If specified, a precomputed weight matrix should be provided.
        It can be computed by either assuming  the independence of noise between conditions,
        or by estimating the noise corvariance structure between conditions.
        If provided as list, the length should match the first dimension of a rdm_array.
        Defaults to None.

        see also:
        Diedrichsen, J. et al. Comparing representational geometries using whitened unbiased-distance-matrix similarity.
        Neuron Behav Data Anal Theory 5, 1–31 (2020).
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to use all available cores.
        Defaults to 1.
    verbose : bool
        If specified, a progress bar appears at the prompt.

    Returns
    -------
    rsa_vals : ndarray, shape ([n_vertices,] [n_times,] [n_rdm_models])
        The RSA value for each data point in input array.
        When multiple models have been supplied, the last dimension will contain RSA results for each model
        (also true for regression-rsq-nnls).
    """
    # reshape rdm_array
    n_rois, n_times, n_dists = rdm_array.shape
    rdm_array = np.reshape(rdm_array, (-1, n_dists)) # to (n_rois * n_times) X n_dists

    # reshape covariance matrix if any
    n_loop = n_rois * n_times
    if isinstance(weights, list) and len(weights) > 1:
        assert len(weights) == n_rois
        weightings = [weights[i] for i in range(n_rois) for j in range(n_times)]
    else:
        weightings = [None for i in range(n_loop)]

    # rdm_model into a condensed form
    if type(rdm_model) == list:
        rdm_model = [_ensure_condensed(rdm, 'rdm_model') for rdm in rdm_model]
    else:
        rdm_model = [_ensure_condensed(rdm_model, 'rdm_model')]

    # define a function for running rsa at a single point
    def rsa_single_rdm(neural_rdm, weighting):
        """
        Compute RSA for at a single roi and time point.

        Input
        ----------
        neural_rdm : ndarray, shape (n_dists,)
            A subset of rdm_array. Specifically, neural rdm at a single roi and time point.
        weighting : ndarray, shape (n_cond, n_cond)
            A weight matrix correspond to neural_rdm (for each roi and time point).
        """
        return _rsa_single_rdm(neural_rdm, rdm_model, rsa_metric, weighting)

    if verbose:
        @contextlib.contextmanager
        def tqdm_joblib(tqdm_object):
            """
            Context manager to patch joblib to report into tqdm progress bar given as argument
            Taken from here:
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

        with tqdm_joblib(tqdm(desc="ROIs", total=len(rdm_array))) as pbar:
            # Call RSA multiple times in parallel for each ROI patch
            data = Parallel(n_jobs)(delayed(rsa_single_rdm)(neural_rdm, weighting)
                                    for neural_rdm, weighting in zip(rdm_array, weightings) )
    else: # without progress bar
        data = Parallel(n_jobs)(delayed(rsa_single_rdm)(neural_rdm, weighting)
                                for neural_rdm, weighting in zip(rdm_array, weightings))

    # Figure out the desired dimensions of the resulting array
    dims = (n_rois, n_times)
    if len(rdm_model) > 1:
        dims = dims + (len(rdm_model),)

    return np.array(data).reshape(dims)


def rsa_stcs_rois(rdm_array, rdm_model, rsa_metric='regression-rsq-nnls', weights=None, n_jobs=1, verbose=True):
    """This function checks the data compatibility before performing RSA."""
    # if a single model RDM, wrap it in list
    one_model = type(rdm_model) is np.ndarray
    if one_model:
        rdm_model = [rdm_model]

    # Check for compatibility of the rdm_array and the model features
    for model in rdm_model:
        if model.ndim == 2:
            model = _ensure_condensed(model, 'rdm_model')
        if rdm_array.shape[-1] != model.shape[0]:
            raise ValueError(
                'The number of distance in rdm_array (%d) should be equal to the '
                'number of distance in `rdm_model` (%d). '
                % (rdm_array.shape[-1], model.shape[0]))

    # Perform the RSA
    data = rsa_array(rdm_array=rdm_array, rdm_model=rdm_model, rsa_metric=rsa_metric,
                     weights=weights, n_jobs=n_jobs, verbose=verbose)

    return data
