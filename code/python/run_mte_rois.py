## run ROI-based mTE
#
# written by S-C. Baek
# update: 16.12.2024
#
'''
The following code is to run multivariate transfer entropy (mTE) analysis a subset of ROIs.

Only the ROIs that shows significant representations for phonemes or prosody were considered in the mTE analysis
to investigate the transfer of the corresponding representations.
'''
import os
import copy
import pickle
import itertools

import numpy as np

from mte_func import compute_mte

from utils import whitening_rdms, get_main_dir

from tqdm import tqdm

# path settings
MAINDIR = get_main_dir()
DATADIR = MAINDIR + os.sep + 'data/'


def run_mte_rois(subjects, element, rois_subset_indices, max_lag=0.3, white=True, n_jobs=-1, verbose=False):
    """
    This function compute raw mTE values from the source to target ROIs based on neural RDM time series.

    Input
    ----------
    subjects : list of str
        List of subject ID in the DATA directory.
    element : str
        A linguistic element to be analysis. it should be either 'phoneme' or 'prosody'.
    rois_subset_indices : ndarray, shape (n_rois_subset,)
        Indices (int) corresponding to the ROIs considered in the analysis.
        Only the ROIs that show significant general representations corresponding to 'element' are included
        in the analysis.
    max_lag : float
        A maximum time lag between the source and target neural RDM time series in second.
        Defaults to 0.3.
    white : bool
        An option to apply whitening to distance estimates while performing mTE analysis.
        If ROI-wise noise covariance across conditions exists, it is employed.
        Otherwise, a whitening matrix is computed based on an identity matrix.
        If None, whitening is not applied.
        Defaults to True.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to use all available cores.
        Defaults to 1.
    verbose : bool
        If specified, a progress bar appears at the prompt.

    Outcome
    ----------
    Save a dictionary 'mte' in individual data directory.
    'mte' contains the following keys.
    SOURCE_to_TARGET : ndarray, shape ([n_time_lags,])
        mTE values from the source to target ROIs at variable time lag.
    max_lag : int
        A maximum time lag in sample.
    n_sample : int
        The number of time samples used for computing individual mTE values.
    """
    # check the input
    if element not in ['phoneme', 'prosody']:
        raise ValueError("Invalid element, must be one of: 'phoneme' or 'prosody' ")

    # loop over subjects
    for subject in subjects:

        # output file
        fname = DATADIR + subject + os.sep + 'meg' + os.sep + 'mte_rois_' + element + '.pickle'
        if os.path.exists(fname): # when the file already exists
            print('%s: existing results of ROI-based mTE for %s ...' % (subject, element))
        else:
            print('%s: run ROI-based mTE for %s ...' %(subject, element))

            # ---------- input files ---------- #
            # import neural RDMs
            print('Import ROI-wise neural RDMs ...')
            frdm_rois = DATADIR + subject + os.sep + 'meg' + os.sep + 'rdm_rois_' + element + '.pickle'
            with open(frdm_rois, 'rb') as f:
                neural_rdms_rois = pickle.load(f)
            rdms = neural_rdms_rois['rdms']
            n_rois, n_times, n_dists = rdms.shape
            n_conds = int(np.ceil(np.sqrt(n_dists * 2)))

            # if white is True
            if white:
                # import noise covaraince matrix across conditions
                print(subject + ': import neural covariance matrix for whitening ...')
                fcov_rois = DATADIR + subject + os.sep + 'meg' + os.sep + 'noise_cov_rois_' + element + '.pickle'
                if not os.path.exists(fcov_rois):  # when no noise cov.
                    print('compute weight matrices based identity matrix, as there is no noise covariance ...')
                    cov = np.eye(n_conds)
                else:
                    with open(fcov_rois, 'rb') as f:
                        neural_cov_rois = pickle.load(f)
                    cov = neural_cov_rois['noise_cov']
                    del neural_cov_rois

            # ---------- parameter specification ---------- #
            # get the information about neural RDMs
            rois = np.array(neural_rdms_rois['labels'])
            tmin = neural_rdms_rois['tmin']
            tstep = neural_rdms_rois['tstep']
            assert n_rois == len(rois)

            # reconstruct time information
            tmax = tmin + tstep * (n_times - 1)
            times = np.linspace(tmin, tmax, n_times)
            del tmin, tmax

            # onset time point
            t0Idx = np.argmin(np.abs(times))

            # only consider data after the stimulus onset
            rdms = rdms[:, t0Idx:, :]
            n_samples = rdms.shape[1]  # n_samples to use

            # max lag into sample
            max_lag_samp = int(max_lag / tstep)
            del neural_rdms_rois, frdm_rois, tstep

            # ---------- main functionality ---------- #
            # consider only a subset of ROIs
            rois_subset_indices = np.array(rois_subset_indices)
            assert len(rois_subset_indices) > 1  # there should be more than one ROI to compute connectivity
            rois_subset = rois[rois_subset_indices]

            # define directed connections
            conn_set = itertools.permutations(rois_subset, 2)
            n_conn = int(len(rois_subset) * (len(rois_subset) - 1))

            # init dict to save data
            mte = dict()

            # loop over combinations
            for i, pair in enumerate(tqdm(conn_set, desc=' Combinations of the ROIs ', total=n_conn)):

                # define the direction of information
                conn = pair[0] + '_to_' + pair[1]

                # find index of the rois in analysis
                source_idx = np.where(pair[0] == rois)[0][0]
                target_idx = np.where(pair[1] == rois)[0][0]

                # fetch the corresponding data
                source = rdms[source_idx]
                target = rdms[target_idx]

                # whitening the rdms if specified
                if white and isinstance(cov, list):
                    source_cov = cov[source_idx]
                    assert n_conds == source_cov.shape[0]
                    assert source_cov.shape[0] == source_cov.shape[1]
                    source = whitening_rdms(source, source_cov)

                    target_cov = cov[target_idx]
                    assert n_conds == target_cov.shape[0]
                    assert target_cov.shape[0] == target_cov.shape[1]
                    target = whitening_rdms(target, target_cov)
                    del source_cov, target_cov
                elif white and isinstance(cov, np.ndarray):
                    assert cov.ndim == 2
                    assert n_conds == cov.shape[0]
                    assert cov.shape[0] == cov.shape[1]
                    target = whitening_rdms(target, cov)
                    source = whitening_rdms(source, cov)

                # compute multivariate transfer entropy
                te = compute_mte(target=target, source=source, max_lag=max_lag_samp, n_jobs=n_jobs, verbose=verbose)
                mte[conn] = te
                del conn, te, target, target_idx, source, source_idx

            # store additional info.
            mte['max_lag'] = max_lag_samp
            mte['n_samples'] = n_samples

            # save rsa_rois
            with open(fname, 'wb') as f:
                pickle.dump(mte, f)
            f.close()
            del mte, fname, rdms, cov


def run_mte_rois_null(subjects, element, rois_subset_indices, max_lag=0.3, white=True,
                      n_perms=200, n_jobs=-1, verbose=False):
    """
    This function compute null mTE values from the source to target ROIs based on neural RDM time series.
    The null mTE values are derived from cyclically time-shifting the source time series.

    Input
    ----------
    subjects : list of str
        List of subject ID in the DATA directory.
    element : str
        A linguistic element to be analysis. it should be either 'phoneme' or 'prosody'.
    rois_subset_indices : ndarray, shape (n_rois_subset,)
        Indices (int) corresponding to the ROIs considered in the analysis.
        Only the ROIs that show significant general representations corresponding to 'element' are included
        in the analysis.
    max_lag : float
        A maximum time lag between the source and target neural RDM time series in second.
        Defaults to 0.3.
    white : bool
        An option to apply whitening to distance estimates while performing mTE analysis.
        If ROI-wise noise covariance across conditions exists, it is employed.
        Otherwise, a whitening matrix is computed based on an identity matrix.
        If None, whitening is not applied.
        Defaults tp True.
    n_perms : int
        The number of permutations. During permutations, the source time series is cyclically time-shifted
        based on a break point randomly sampled from the interval of n_times*[0.05, 0.95].
        If n_perm is larger than the number of samples within the interval n_times*[0.05, 0.95],
        all possible break points within the interval are used.
        Defaults to 200.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to use all available cores.
        Defaults to 1.
    verbose : bool
        If specified, a progress bar appears at the prompt.

    Outcome
    ----------
    Save a dictionary 'mte_null' in individual data directory.
    'mte_null' contains the following keys.
    SOURCE_to_TARGET : ndarray, shape ([n_time_lags,] [n_perm,])
        null mTE values from the source to target ROIs at variable time lags in given permutation.
    max_lag : int
        A maximum time lag in sample.
    n_sample : int
        The number of time samples used for computing individual mTE values.
    """
    # check the input
    if element not in ['phoneme', 'prosody']:
        raise ValueError("Invalid element, must be one of: 'phoneme' or 'prosody' ")

    # loop over subjects
    for subject in subjects:

        # output file
        fname = DATADIR + subject + os.sep + 'meg' + os.sep + 'mte_rois_' + element + '_null.pickle'
        if os.path.exists(fname):  # when the file already exists
            print('%s: existing results of ROI-based mTE (null) for %s ...' % (subject, element))
        else:
            print('%s: run ROI-based mTE (null) for %s ...' % (subject, element))

            # ---------- input files ---------- #
            # import neural RDMs
            print('Import ROI-wise neural RDMs ...')
            frdm_rois = DATADIR + subject + os.sep + 'meg' + os.sep + 'rdm_rois_' + element + '.pickle'
            with open(frdm_rois, 'rb') as f:
                neural_rdms_rois = pickle.load(f)
            rdms = neural_rdms_rois['rdms']
            n_rois, n_times, n_dists = rdms.shape
            n_conds = int(np.ceil(np.sqrt(n_dists * 2)))

            # if white is True
            if white:
                # import noise covaraince matrix across conditions
                print(subject + ': import neural covariance matrix for whitening ...')
                fcov_rois = DATADIR + subject + os.sep + 'meg' + os.sep + 'noise_cov_rois_' + element + '.pickle'
                if not os.path.exists(fcov_rois):  # when no noise cov.
                    print('compute weight matrices based identity matrix, as there is no noise covariance ...')
                    cov = np.eye(n_conds)
                else:
                    with open(fcov_rois, 'rb') as f:
                        neural_cov_rois = pickle.load(f)
                    cov = neural_cov_rois['noise_cov']
                    del neural_cov_rois

            # ---------- parameter specification ---------- #
            # get the information about neural RDMs
            rois = np.array(neural_rdms_rois['labels'])
            tmin = neural_rdms_rois['tmin']
            tstep = neural_rdms_rois['tstep']
            assert n_rois == len(rois)

            # reconstruct time information
            tmax = tmin + tstep * (n_times - 1)
            times = np.linspace(tmin, tmax, n_times)
            del tmin, tmax

            # onset time point
            t0Idx = np.argmin(np.abs(times))

            # only consider data after the stimulus onset
            rdms = rdms[:, t0Idx:, :]
            n_samples = rdms.shape[1]  # n_samples to use

            # max lag into sample
            max_lag_samp = int(max_lag / tstep)
            lags = np.arange(1, max_lag_samp + 1)
            n_lags = len(lags)
            del neural_rdms_rois, frdm_rois, tstep

            # ---------- determine permutations ---------- #
            perm_bound = np.ceil(n_samples * np.array([0.05, 0.95])).astype(np.intc)
            perm_range = np.arange(perm_bound[0], perm_bound[-1])
            n_possible_perm = len(perm_range)
            if n_perms > n_possible_perm:
                print(' A specified input (%s) exceeds the number of possible permutations ... ' % n_perms)
                print(' Perform exact permutations (n_perm = %s) ... ' % n_possible_perm)
                perms = copy.deepcopy(perm_range)
            else:
                print(' prepare the list of random permutations ... ')
                perms = np.random.choice(perm_range, n_perms, replace=False)
            n_perms = len(perms)

            # ---------- determine permutations ---------- #
            # consider only a subset of ROIs
            rois_subset_indices = np.array(rois_subset_indices)
            assert len(rois_subset_indices) > 1  # there should be more than one ROI to compute connectivity
            rois_subset = rois[rois_subset_indices]

            # define directed connections
            conn_set = itertools.permutations(rois_subset, 2)
            n_conn = int(len(rois_subset) * (len(rois_subset) - 1))

            # init dict to save data
            mte_null = dict()

            # loop over combinations
            for i, pair in enumerate(tqdm(conn_set, desc=' Combinations of the ROIs ', total=n_conn)):

                # define the direction of information
                conn = pair[0] + '_to_' + pair[1]

                # find index of the rois in analysis
                source_idx = np.where(pair[0] == rois)[0][0]
                target_idx = np.where(pair[1] == rois)[0][0]

                # fetch the corresponding data
                source = rdms[source_idx]
                target = rdms[target_idx]

                # whitening the rdms if specified
                if white and isinstance(cov, list):
                    source_cov = cov[source_idx]
                    assert n_conds == source_cov.shape[0]
                    assert source_cov.shape[0] == source_cov.shape[1]
                    source = whitening_rdms(source, source_cov)

                    target_cov = cov[target_idx]
                    assert n_conds == target_cov.shape[0]
                    assert target_cov.shape[0] == target_cov.shape[1]
                    target = whitening_rdms(target, target_cov)
                    del source_cov, target_cov
                elif white and isinstance(cov, np.ndarray):
                    assert cov.ndim == 2
                    assert n_conds == cov.shape[0]
                    assert cov.shape[0] == cov.shape[1]
                    target = whitening_rdms(target, cov)
                    source = whitening_rdms(source, cov)

                # compute multivariate transfer entropy with permuted data
                te_null = np.zeros((n_lags, n_perms))
                for p in range(n_perms):
                    # permute source time series only
                    source_null = np.concatenate([source[perms[p]:], source[:perms[p]]])
                    te_null[:, p] = compute_mte(target=target, source=source_null, max_lag=max_lag_samp,
                                                n_jobs=n_jobs, verbose=verbose)
                mte_null[conn] = te_null
                del conn, te_null, target, target_idx, source, source_idx, source_null

            # additional info.
            mte_null['max_lag'] = max_lag_samp
            mte_null['n_samples'] = rdms.shape[1]

            # save rsa_rois
            with open(fname, 'wb') as f:
                pickle.dump(mte_null, f)
            f.close()
            del mte_null, fname, rdms, cov


def normalize_mte(subjects, element):
    """
    This function normalizes raw mTE values based on the mean and standard deviation of the null mTE distribution.

    Input
    ----------
    subjects : list of str
        List of subject ID in the DATA directory.
    element : str
        A linguistic element to be analysis. it should be either 'phoneme' or 'prosody'.

    Outcome
    ----------
    Save a dictionary 'mte_norm' in individual data directory.
    'mte_norm' contains the following keys.
    SOURCE_to_TARGET : ndarray, shape ([n_time_lags,])
        mTE values normalized by the mean and standard deviation of the null mTE distribution at a given directed
        connection and time lag (corresponding to Z scores).
    max_lag : int
        A maximum time lag in sample.
    n_sample : int
        The number of time samples used for computing individual mTE values.
    """
    # check the input
    if element not in ['phoneme', 'prosody']:
        raise ValueError("Invalid element, must be one of: 'phoneme' or 'prosody' ")

    # loop over subjects
    for subject in subjects:

        # output file
        fname = DATADIR + subject + os.sep + 'meg' + os.sep + 'mte_rois_' + element + '_norm.pickle'
        if os.path.exists(fname):  # when the file already exists
            print('%s: existing results of ROI-based mTEn (Z score) for %s ...' % (subject, element))
        else:
            print('%s: run ROI-based mTEn (Z score) for %s ...' % (subject, element))

            # ---------- input files ---------- #
            # import raw mTE
            print('Import ROI-wise mTE values (raw) ...')
            fmte_raw = DATADIR + subject + os.sep + 'meg' + os.sep + 'mte_rois_' + element + '.pickle'
            with open(fmte_raw, 'rb') as f:
                mte = pickle.load(f)
            conn = np.array([c for c in mte.keys() if '_to_' in c])
            n_conns = len(conn)

            # import null mTE
            print('Import ROI-wise mTE values (null) ...')
            fmte_null = DATADIR + subject + os.sep + 'meg' + os.sep + 'mte_rois_' + element + '_null.pickle'
            with open(fmte_null, 'rb') as f:
                mte_null = pickle.load(f)

            # ---------- parameter specification ---------- #
            n_lags, n_perms = mte_null[conn[0]].shape

            # ---------- main functionality ---------- #
            # init dict to save data
            mte_norm = dict()

            # loop over combinations
            for i, c in enumerate(tqdm(conn, desc=' Combinations of the ROIs ', total=n_conns)):

                # get the observations corresponding to a directed connection
                te = mte[c]
                assert len(te) == n_lags

                # get the null te based on the permuted data
                null_te = mte_null[c]
                m_null_te = np.mean(null_te, axis=-1) # mean of permutations
                sd_null_te = np.std(null_te, axis=-1) # std of permutations

                # normalize te
                te_norm = (te - m_null_te) / sd_null_te
                mte_norm[c] = te_norm
                del te, null_te, m_null_te, sd_null_te, te_norm

            # additional info.
            mte_norm['max_lag'] = mte['max_lag']
            mte_norm['n_samples'] = mte['n_samples']

            # save rsa_rois
            with open(fname, 'wb') as f:
                pickle.dump(mte_norm, f)
            f.close()
            del mte, fmte_raw, mte_null, fmte_null, mte_norm, fname