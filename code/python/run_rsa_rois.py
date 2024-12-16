## run ROI-based RSA
#
# written by S-C. Baek
# update: 16.12.2024
#
'''
The following code is to run representational similarity analysis (RSA) on 10 predefined ROIs or a subset of them.

Here, RSA basically measures the similarity between each or time-resolved neural RDMs at a given ROI and time point,
and two model RDMs (AcoustRDM & CatRDM), separately for phonemes and prosody.

Not only task-collapsed data but also task-specific data are considered
to perform for the analysis of task-modulation effect.
'''
import os
import pickle

import numpy as np
import pandas as pd

from rsa_func import rsa_stcs_rois

from utils import get_v, get_main_dir

# path settings
MAINDIR = get_main_dir()
DATADIR = MAINDIR + os.sep + 'data/'


def _is_twindow_applied(tbeg, tmin, tstep, twindow):
    """This function checks if time window is applied when computing neural RDMs."""
    return np.abs(tmin-tbeg-twindow/2) < tstep


def _apply_twindow(rdms, tmin, tstep, twindow):
    """
    This function applied time windows to sample-wise neural RDMs.
    Note that the order of applying time windows and computing Euclidean distance can be swapped,
    because both are linear operations.
    """
    # parameters
    n_rois, n_times, n_dists = rdms.shape

    # time window into samples
    twindow_samp = int(twindow/tstep)
    half_twindow_samp = int(twindow_samp/2)

    # update time information
    tmin = tmin + half_twindow_samp*tstep
    tmin_samp = half_twindow_samp
    tmax_samp = n_times - half_twindow_samp
    n_times -= twindow_samp
    tcenter_samp = np.arange(tmin_samp, tmax_samp) # center points of windows in sample

    # apply time windows to rdms
    rdms_windowed = np.zeros( (n_rois, n_times, n_dists) )
    for roii in range(n_rois):
        for ti, samp in enumerate(tcenter_samp):
            rdms_windowed[roii,ti,:]=np.mean(rdms[roii, samp-half_twindow_samp:samp+half_twindow_samp+1, :], axis=0)

    return rdms_windowed, tmin


def run_rsa_rois(subjects, element, tbeg=-0.2, twindow=0.024, gen=True, task='', white=True, n_jobs=-1, verbose=True):
    """
    This function performs ROI-based RSA in a time-resolved manner.

    Input
    ----------
    subjects : list of str
        List of subject ID in the DATA directory.
    element : str
        A linguistic element to be analysis. it should be either 'phoneme' or 'prosody'.
    tbeg : float
        An initial time point of epoch. It is used to check neural RDMs were computed within time windows.
        The original neural RDMs were computed within time windows, but here we used sample-wise neural RDMs
        to reduce the volume of data shared.
        Note that time windowed applied before and after computing neural RDMs (Euclidean dist.)
        output the same results, as both are linear operations.
        Defaults to -0.2.
    twindow : float
        The size of time windows applied in second.
        Defaults to 0.024.
    gen : bool
        If true, general representations (Acoust+CatRDM) is computed.
        Otherwise, unique acoustic and categorical representations are computed.
        Defaults to True.
    task : str
        An option to perform RSA on task-specific data.
        It should be specified between 'att' (relevant task to element) and 'ign' (irrelevant task to element).
        Defaults to ''.
    white : bool
        An option to apply whitening to distance estimates while performing RSA.
        If ROI-wise noise covariance across conditions exists, it is employed.
        Otherwise, a whitening matrix is computed based on an identity matrix.
        If None, whitening is not applied.
        Defaults tp True.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to use all available cores.
        Defaults to 1.
    verbose : bool
        If specified, a progress bar appears at the prompt.

    Outcome
    ----------
    Save a dictionary 'rsa_rois' in individual data directory.
    'rsa_rois' contains the following keys.
    space : str
        Cortical surface where the analysis was implemented (here, fsaverage for all participants).
    labels : list of str
        List of ROI labels (e.g., L-PAC, etc.).
    roi_inds : list of ndarray
        List of arrays that represent the vertices on 'space' included for defining ROIs.
    tmin : float
        An initial time point of rsa_vals.
    tstep : float
        The size of step for time windows in second.
    rsa_vals : ndarray, shape ([n_vertices,] [n_times,]) | ([n_vertices,] [n_times,] [n_rdm_models])
        Time-resolved RSA values for ROIs.
        The last dimension corresponds to the number of model RDMs considered (only for gen==False).
    """
    # check the input
    if element not in ['phoneme', 'prosody']:
        raise ValueError("Invalid element, must be one of: 'phoneme' or 'prosody' ")
    if task and task not in ['att', 'ign']:
        raise ValueError("Invalid task, must be one of: 'att' or 'ign' ")

    # loop over subjects
    for subject in subjects:

        # output file
        fname = DATADIR + subject + os.sep + 'meg' + os.sep + 'rsa_rois_' + element
        fname += '_gen' if gen else '_uni'
        fname += '_' + task if task else ''
        fname += '.pickle'
        if os.path.exists(fname): # when the file already exists
            print('%s: existing results of ROI-based RSA (%s) for %s%s ...'
                  % (subject, 'gen' if gen else 'uni', element, ' ('+task+')' if task else ''))
        else:
            print('%s: run ROI-based RSA (%s) for %s%s ...'
                  % (subject, 'gen' if gen else 'uni', element, ' ('+task+')' if task else ''))

            # ---------- input files ---------- #
            # import neural RDMs
            print('Import ROI-wise neural RDMs ...')
            frdm_rois = DATADIR + subject + os.sep + 'meg' + os.sep + 'rdm_rois_' + element
            frdm_rois += '_' + task if task else ''
            frdm_rois += '.pickle'
            with open(frdm_rois, 'rb') as f:
                neural_rdms_rois = pickle.load(f)
            n_rois, _, n_dists = neural_rdms_rois['rdms'].shape

            # import model RDMs
            print('Import model RDMs ...')
            frdm_model = DATADIR + subject + os.sep + 'behavior' + os.sep + 'rdm_models_' + element + '.txt'
            df = pd.read_csv(frdm_model, sep='\t', index_col=0)
            n_conds = int(np.sqrt(len(df.index)))  # number of conditions
            upper = np.triu_indices(n_conds, k=1)  # indices of upper off-diagonal entries
            model_rdms = [df.values[:, idx].reshape(n_conds, -1)[upper] for idx, _ in enumerate(df.columns)]
            del df, upper

            # if white is True
            if white:
                # list to pack weight matrices
                Vs = list()

                # import noise covaraince matrix across conditions
                print(subject + ': import neural covariance matrix for whitening ...')
                fcov_rois = DATADIR + subject + os.sep + 'meg' + os.sep + 'noise_cov_rois_' + element + '.pickle'
                if not os.path.exists(fcov_rois): # when no noise cov.
                    print('compute weight matrices based identity matrix, as there is no noise covariance ...')
                    for roii in range(n_rois):
                        Vs.append(get_v(n_conds, sigma_k=None))
                else:
                    with open(fcov_rois, 'rb') as f:
                        neural_cov_rois = pickle.load(f)
                    for roii in range(n_rois):
                        Vs.append(get_v(n_conds, sigma_k=neural_cov_rois['noise_cov'][roii]))
                    del neural_cov_rois
            else: # if not, no weight matrices
                Vs = None

            # ---------- post-processing to neural RDMs ---------- #
            '''Note: the original neural rdms were based time-windowed data.
                     Applying time windows later is to reduce the volume of dataset shared. '''
            # when time windows were already applied
            if _is_twindow_applied(tbeg, neural_rdms_rois['tmin'], neural_rdms_rois['tstep'], twindow):
                rdms = neural_rdms_rois['rdms']
                tmin = neural_rdms_rois['tmin']
            else: # when time windows are not applied in advance
                # apply time windows to sample-wise neural RDMs
                rdms, tmin = _apply_twindow(neural_rdms_rois['rdms'],
                                            neural_rdms_rois['tmin'],
                                            neural_rdms_rois['tstep'],
                                            twindow)
            tstep = neural_rdms_rois['tstep']

            # ---------- rsa metric depending on 'gen' ---------- #
            rsa_metric = 'regression-rsq-nnls' if gen else 'nested-comp-uvar-nnls'

            # ---------- main functionality ---------- #
            rsa_vals = rsa_stcs_rois(rdm_array=rdms,
                                     rdm_model=model_rdms,
                                     rsa_metric=rsa_metric,
                                     weights=Vs,
                                     n_jobs=n_jobs,
                                     verbose=verbose)
            if gen:
                rsa_vals = rsa_vals[:,:,0] # 0 and 1 are the same for general represenatations
                assert rsa_vals.ndim == 2

            # ---------- save data ---------- #
            # results into dictionary
            rsa_rois = dict(space=neural_rdms_rois['space'],
                            labels=neural_rdms_rois['labels'],
                            roi_inds=neural_rdms_rois['roi_inds'],
                            tmin=tmin,
                            tstep=tstep,
                            rsa_vals=rsa_vals)
            del neural_rdms_rois

            # save rsa_rois
            with open(fname, 'wb') as f:
                pickle.dump(rsa_rois, f)
            f.close()
            del rsa_rois