## cluster statistics
#
# written by S-C. Baek
# update: 16.12.2024
#
'''
The following code is to infer the results of time-resolved RSA or mTE analysis at the group-level.
'''
import os
import pickle
import itertools

import numpy as np

from scipy import stats as stats

import mne

from utils import fisher_z_transform, get_main_dir

# path settings
MAINDIR = get_main_dir()
DATADIR = MAINDIR + os.sep + 'data/'


def cluster_1samp_ttest_rsa(subjects, element, testvar, n_permutations=100000,
                            p_threshold=0.05, tail=1, n_jobs=-1, verbose=True):
    """
    This function infer time-resolved RSA in 10 ROIs at the group-level.

    Input
    ----------
    subjects : list of str
        List of subject ID in the DATA directory.
    element : str
        A linguistic element to be analysis. it should be either 'phoneme' or 'prosody'.
    testvar : str
        variable to test. It should be either 'gen' or 'uni'.
        If gen, general representations are tested.
        Otherwise, unique acoustic and categorical representations are tested, separately.
    n_permutations : int
        The number of permutations implemented for null distribution.
        Defaults to 100000.
    p_threshold : float
        For a cluster-forming threshold at p < p_threshold.
        Defaults to 0.05.
    tail : int
        A value to determine one-sided (1) or two-sided (0) test.
        In case tail = -1, one-sided test in a negative direction.
        Defaults to 1.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to use all available cores.
        Defaults to 1.
    verbose : bool
        If specified, a progress bar appears at the prompt.

    Outcome
    ----------
    Save 'cluster extent' and 'cluster p-values' for each ROI (uncorrected)
    """
    # check the input
    if element not in ['phoneme', 'prosody']:
        raise ValueError("Invalid element, must be one of: 'phoneme' or 'prosody' ")
    if testvar not in ['gen', 'uni']:
        raise ValueError("Invalid testing variable, must be one of: 'gen' or 'uni' ")

    # output file
    fpath = DATADIR + 'group' + os.sep + 'meg' + os.sep
    fname = fpath + 'rsa_rois_' + element + '_' + testvar + '_vs_baseline_p' + str(p_threshold)[2:] + '.pickle'
    if os.path.exists(fname):  # when the file already exists
        print('Existing statistical results of RSA (%s) for %s ...' % (testvar, element))
    else:
        print('Run statistical inference of RSA (%s) for %s ...' % (testvar, element))

        # ---------- parameter specification from template data ---------- #
        ftmp = DATADIR + subjects[0] + os.sep + 'meg' + os.sep + 'rsa_rois_' + element + '_' + testvar + '.pickle'
        with open(ftmp, 'rb') as f:
            tmp = pickle.load(f)
        n_rois, n_times = tmp['rsa_vals'].shape[:2]
        n_models = 1 if tmp['rsa_vals'].ndim == 2 else 2
        rois = np.array(tmp['labels'])

        # reconstruct the time
        tmin = tmp['tmin']
        tstep = tmp['tstep']
        n_times = tmp['rsa_vals'].shape[1]
        tmax = tmin + tstep * (n_times - 1)
        times = np.linspace(tmin, tmax, n_times)
        del tmp, ftmp

        # ---------- import data ---------- #
        # init a matrix into which all individual data will be imported
        n_subjects = len(subjects)
        X = np.zeros((n_rois, n_times, n_models, n_subjects))

        # loop over subjects
        for subi, subject in enumerate(subjects):
            # import individual rsa_vals
            frsa = DATADIR + subject + os.sep + 'meg' + os.sep + 'rsa_rois_' + element + '_' + testvar + '.pickle'
            with open(frsa, 'rb') as f:
                rsa = pickle.load(f)
            X[:, :, :, subi] = rsa['rsa_vals'][:,:,np.newaxis] if n_models == 1 else rsa['rsa_vals']
            del rsa, frsa

        # apply Fisher Z transformation to R^2 after taking square root of it
        X = fisher_z_transform(X, sqrt=True)

        # apply baseline correction to the data
        t0Idx = np.argmin(np.abs(times))  # onset index
        Z = X - np.mean(X[:, :t0Idx, :, :], axis=1)[:, np.newaxis, :, :]  # baseline correction
        del X

        # ---------- main functionality ---------- #
        # cluster forming threshold
        if tail == 1:
            t_threshold = stats.distributions.t.ppf(1 - p_threshold, df=n_subjects - 1)
            msg = 'one-tailed t-test with t-threshold of %.3f (p=%.3f)' % (t_threshold, p_threshold)
        elif tail == 0:
            t_threshold = stats.distributions.t.ppf(1 - p_threshold/2, df=n_subjects - 1)
            msg = 'two-tailed t-test with t-threshold of %.3f (p=%.3f)' % (t_threshold, p_threshold)
        print(msg)

        # loop over models
        cluster_stats_models = list()
        for i in range(n_models):

            # data to test
            z = Z[:, :, i, :]  # now, to n_rois X n_times X n_subjects

            # loop over rois for testing
            cluster_stats_rois = list()
            for j in range(len(z)):
                print('Testing %s#%d of %s in %s' % (testvar, i+1, element, rois[j]))
                data = z[j, t0Idx:, :]  # now, to n_times X n_subjects; from the onset
                data = data.T  # to n_subject X n_times

                # Now let's actually do the clustering. This can take a long time...
                _, clusters, cluster_p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(
                    data,
                    adjacency=None,  # make the ROIs independent
                    max_step=1,
                    tail=tail,
                    threshold=t_threshold,
                    n_permutations=n_permutations,
                    n_jobs=n_jobs,
                    buffer_size=None,
                    verbose=verbose)

                # put clustering results in to the list
                cluster_stats = (clusters, cluster_p_values)
                cluster_stats_rois.append(cluster_stats)

            # put model-wise results into a bigger list
            cluster_stats_models.append(cluster_stats_rois)

        # peel off list if n_models == 1
        if len(cluster_stats_models) == 1:
            cluster_stats_models = cluster_stats_models[0]

        # save cluster statistics
        with open(fname, 'wb') as f:
            pickle.dump(cluster_stats_models, f)
        f.close()


def cluster_1samp_ttest_rsa_task_modul(subjects, element, testvar, rois_subset_indices,
                                       n_permutations=100000, p_threshold=0.05, tail=1, n_jobs=-1, verbose=True):
    """
    This function infer task-modulation effect on time-resolved RSA in a subset of ROIs at the group-level.

    Input
    ----------
    subjects : list of str
        List of subject ID in the DATA directory.
    element : str
        A linguistic element to be analysis. it should be either 'phoneme' or 'prosody'.
    testvar : str
        variable to test. It should be either 'gen' or 'uni'.
        If gen, general representations are tested.
        Otherwise, unique acoustic and categorical representations are tested, separately.
    rois_subset_indices : ndarray, shape (n_rois_subset,)
        Indices (int) corresponding to the ROIs considered in the analysis.
        Only the ROIs that show significant general representations corresponding to 'element' are included in the analysis.
    n_permutations : int
        The number of permutations implemented for null distribution.
        Defaults to 100000.
    p_threshold : float
        For a cluster-forming threshold at p < p_threshold.
        Defaults to 0.05.
    tail : int
        A value to determine one-sided (1) or two-sided (0) test.
        In case tail = -1, one-sided test in a negative direction.
        Defaults to 1.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to use all available cores.
        Defaults to 1.
    verbose : bool
        If specified, a progress bar appears at the prompt.

    Outcome
    ----------
    Save 'cluster extent' and 'cluster p-values' for each ROI (uncorrected)
    """
    # check the input
    if element not in ['phoneme', 'prosody']:
        raise ValueError("Invalid element, must be one of: 'phoneme' or 'prosody' ")
    if testvar not in ['gen', 'uni']:
        raise ValueError("Invalid testing variable, must be one of: 'gen' or 'uni' ")
    assert isinstance(rois_subset_indices, list) or  isinstance(rois_subset_indices, np.ndarray)
    rois_subset_indices = np.array(rois_subset_indices)
    assert rois_subset_indices.ndim == 1 # should be 1d array

    # output file
    fpath = DATADIR + 'group' + os.sep + 'meg' + os.sep
    fname = fpath + 'rsa_rois_' + element + '_' + testvar + '_att_vs_ign_p' + str(p_threshold)[2:] + '.pickle'
    if os.path.exists(fname):  # when the file already exists
        print('Existing statistical results of task-modulation effect on RSA (%s) for %s ...' % (testvar, element))
    else:
        print('Run statistical inference of task-modulation effect on RSA (%s) for %s ...' % (testvar, element))

        # ---------- parameter specification from template data ---------- #
        ftmp = DATADIR + subjects[0] + os.sep + 'meg' + os.sep + 'rsa_rois_' + element + '_' + testvar + '_att.pickle'
        with open(ftmp, 'rb') as f:
            tmp = pickle.load(f)
        n_rois, n_times = tmp['rsa_vals'].shape[:2]
        n_models = 1 if tmp['rsa_vals'].ndim == 2 else 2
        rois = np.array(tmp['labels'])

        # reconstruct the time
        tmin = tmp['tmin']
        tstep = tmp['tstep']
        n_times = tmp['rsa_vals'].shape[1]
        tmax = tmin + tstep * (n_times - 1)
        times = np.linspace(tmin, tmax, n_times)
        del tmp, ftmp

        # ---------- import data ---------- #
        # init a matrix into which all individual data will be imported
        n_subjects = len(subjects)
        X = np.zeros((n_rois, n_times, n_models, n_subjects))
        Y = np.zeros((n_rois, n_times, n_models, n_subjects))

        # loop over subjects
        for subi, subject in enumerate(subjects):
            # import individual rsa_vals during the relevant task
            frsa = DATADIR + subject + os.sep + 'meg' + os.sep + 'rsa_rois_' + element + '_' + testvar + '_att.pickle'
            with open(frsa, 'rb') as f:
                rsa = pickle.load(f)
            X[:, :, :, subi] = rsa['rsa_vals'][:,:,np.newaxis] if n_models == 1 else rsa['rsa_vals']
            del rsa, frsa

            # import individual rsa_vals during the irrelevant task
            frsa = DATADIR + subject + os.sep + 'meg' + os.sep + 'rsa_rois_' + element + '_' + testvar + '_ign.pickle'
            with open(frsa, 'rb') as f:
                rsa = pickle.load(f)
            Y[:, :, :, subi] = rsa['rsa_vals'][:,:,np.newaxis] if n_models == 1 else rsa['rsa_vals']
            del rsa, frsa

        # apply Fisher Z transformation to R^2 after taking square root of it
        X = fisher_z_transform(X, sqrt=True)
        Y = fisher_z_transform(Y, sqrt=True)

        # apply baseline correction to the data
        t0Idx = np.argmin(np.abs(times))  # onset index
        X = X - np.mean(X[:, :t0Idx, :, :], axis=1)[:, np.newaxis, :, :]  # baseline correction
        Y = Y - np.mean(Y[:, :t0Idx, :, :], axis=1)[:, np.newaxis, :, :]  # baseline correction

        # take the difference between X and Y
        Z = X - Y
        del X, Y

        # ---------- main functionality ---------- #
        # cluster forming threshold
        if tail == 1:
            t_threshold = stats.distributions.t.ppf(1 - p_threshold, df=n_subjects - 1)
            msg = 'one-tailed t-test with t-threshold of %.3f (p=%.3f)' % (t_threshold, p_threshold)
        elif tail == 0:
            t_threshold = stats.distributions.t.ppf(1 - p_threshold/2, df=n_subjects - 1)
            msg = 'two-tailed t-test with t-threshold of %.3f (p=%.3f)' % (t_threshold, p_threshold)
        print(msg)

        # loop over models
        print('Testing %d ROIs out of %d' % (n_rois, len(rois_subset_indices)))
        cluster_stats_models = list()
        for i in range(n_models):

            # data to test
            z = Z[:, :, i, :]  # now, to n_rois X n_times X n_subjects

            # loop over rois for testing
            cluster_stats_rois = list()
            for j in rois_subset_indices:
                print('Testing %s#%d of %s in %s' % (testvar, i+1, element, rois[j]))
                data = z[j, t0Idx:, :]  # now, to n_times X n_subjects; from the onset
                data = data.T  # to n_subject X n_times

                # Now let's actually do the clustering. This can take a long time...
                _, clusters, cluster_p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(
                    data,
                    adjacency=None,  # make the ROIs independent
                    max_step=1,
                    tail=tail,
                    threshold=t_threshold,
                    n_permutations=n_permutations,
                    n_jobs=n_jobs,
                    buffer_size=None,
                    verbose=verbose)

                # put clustering results in to the list
                cluster_stats = (clusters, cluster_p_values)
                cluster_stats_rois.append(cluster_stats)

            # put model-wise results into a bigger list
            cluster_stats_models.append(cluster_stats_rois)

        # peel off list if n_models == 1
        if len(cluster_stats_models) == 1:
            cluster_stats_models = cluster_stats_models[0]

        # save cluster statistics
        with open(fname, 'wb') as f:
            pickle.dump(cluster_stats_models, f)
        f.close()


def cluster_1samp_ttest_mte(subjects, element, n_permutations=100000, p_threshold=0.05, tail=1, n_jobs=-1, verbose=True):
    """
    This function infer representational transfer (mTE) at the group-level.

    Input
    ----------
    subjects : list of str
        List of subject ID in the DATA directory.
    element : str
        A linguistic element to be analysis. it should be either 'phoneme' or 'prosody'.
    n_permutations : int
        The number of permutations implemented for null distribution.
        Defaults to 100000.
    p_threshold : float
        For a cluster-forming threshold at p < p_threshold.
        Defaults to 0.05.
    tail : int
        A value to determine one-sided (1) or two-sided (0) test.
        In case tail = -1, one-sided test in a negative direction.
        Defaults to 1.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to use all available cores.
        Defaults to 1.
    verbose : bool
        If specified, a progress bar appears at the prompt.

    Outcome
    ----------
    Save a dictionary 'cluster_stats_conns' in group data directory.
    'cluster_stats_conns' contains the following keys.
    connections : list
        List of directed connections in the form of 'SOURCE_to_TARGET'.
    sig_clusters_uncorr : int
        List of cluster extexts corresponding to 'connections' (cluster threshold of 0.05 not applied ...)
    sig_p_vals_uncorr : int
        List of cluster p-values corresponding to 'sig_clusters_uncorr' (cluster threshold of 0.05 not applied ...)
    """
    # check the input
    if element not in ['phoneme', 'prosody']:
        raise ValueError("Invalid element, must be one of: 'phoneme' or 'prosody' ")

    # output file
    fpath = DATADIR + 'group' + os.sep + 'meg' + os.sep
    fname = fpath + 'mte_rois_' + element + '_mTEn_vs_0_p' + str(p_threshold)[2:] + '.pickle'
    if os.path.exists(fname):  # when the file already exists
        print('Existing statistical results of mTE anlysis for %s ...' % element)
    else:
        print('Run statistical inference of mTE anlysis for %s ...' % element)

        # ---------- parameter specification from template data ---------- #
        # import template data
        ftmp = DATADIR + subjects[0] + os.sep + 'meg' + os.sep + 'mte_rois_' + element + '_norm.pickle'
        with open(ftmp, 'rb') as f:
            tmp = pickle.load(f)
        conn = [c for c in tmp.keys() if '_to_' in c]
        n_conn = len(conn)
        n_lags = len(tmp[conn[0]])
        n_subjects = len(subjects)
        del tmp, ftmp

        # reconstruct the ROIs
        rois_subset = list()
        for c in conn:
            rs = c.split('_to_')
            for r in rs:
                rois_subset.append(r)
        rois_subset = list(set(rois_subset))
        rois_subset.sort()

        # ---------- main functionality ---------- #
        # a set of connections
        conn_set = itertools.permutations(rois_subset,2)

        # cluster forming threshold
        if tail == 1:
            t_threshold = stats.distributions.t.ppf(1 - p_threshold, df=n_subjects - 1)
            msg = 'one-tailed t-test with t-threshold of %.3f (p=%.3f)' % (t_threshold, p_threshold)
        elif tail == 0:
            t_threshold = stats.distributions.t.ppf(1 - p_threshold/2, df=n_subjects - 1)
            msg = 'two-tailed t-test with t-threshold of %.3f (p=%.3f)' % (t_threshold, p_threshold)
        print(msg)

        # init list
        connections = list()
        sig_clusters_uncorr =list()
        sig_p_vals_uncorr = list()

        # directed connection to test
        for i, (roi_a, roi_b) in enumerate(conn_set):
            print('Testing from %s to %s (%d/%d)' % (roi_a, roi_b, i+1, n_conn))
            c = roi_a + '_to_' + roi_b

            # loop over subjects
            X = np.zeros( (n_lags, n_subjects) )
            for subi, subject in enumerate(subjects):
                # import individual data
                fmte = DATADIR + subject + os.sep + 'meg' + os.sep + 'mte_rois_' + element + '_norm.pickle'
                with open(fmte, 'rb') as f:
                    mte = pickle.load(f)
                X[:, subi] = mte[c]
                del mte,

            # run cluster-based permutation tests
            _, clusters, cluster_p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(
                X.T,  # to n_subjects X n_lags
                adjacency=None,  # make the connections independent
                max_step=1,
                tail=tail,  # one-tail
                threshold=t_threshold,
                n_permutations=n_permutations,
                n_jobs=n_jobs,
                buffer_size=None,
                verbose=verbose)

            # append to pre-defined list
            connections.append(c)
            sig_clusters_uncorr.append(clusters)
            sig_p_vals_uncorr.append(cluster_p_values)
            del c, clusters, cluster_p_values

        # data into dict
        cluster_stats_conns = dict(
            connections=connections,
            sig_clusters_uncorr=sig_clusters_uncorr,
            sig_p_vals_uncorr=sig_p_vals_uncorr
        )

        # save the dict
        with open(fname, 'wb') as f:
            pickle.dump(cluster_stats_conns, f)
        f.close()