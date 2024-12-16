## create model RDMs
#
# written by S-C. Baek
# update: 16.12.2024
#
import os

import scipy.io
from scipy.spatial import distance

import numpy as np
import pandas as pd

from utils import get_main_dir

# path settings
MAINDIR = get_main_dir()
DATADIR = MAINDIR + os.sep + 'data/'


def create_model_rdms(subjects, element):
    '''
    The following code is to create the model representational dissimilary matrices (RDMs),
    separately for phonemes and prosody.

    Specifically, two different model RDMs are generated for each participant and linguistic element:
        acoustic (AcoustRDM) and categorical (CatRDM) RDMs.

    AcoustRDMs are based on the original morph steps from the 61-step continuum of phonemes or prosody.

    CatRDMs are based on the expected proportion of "Paar" or "Question" responses at the five morph levels,
    derived from the sigmoid functions fitted to speaker-collapesed behavioral responses.

    Input
    ----------
    subjects : list of str
        List of subject ID in the DATA directory.
    element : str
        A linguistic element to be analysis. it should be either 'phoneme' or 'prosody'.

    Outcome
    ----------
    Save AcoustRDM and CatRDM for a specified linguistic element as pandas DataFrame.
    (in the following directory: 'MAINDIR/data/sub-??/behavior/')
    Both model RDMs will be stored in a square form (see `scipy.spatial.distance.squareform`),
    but will be flattened such that each model RDM takes one column in the DataFrame.
    '''
    # check the input
    if element not in ['phoneme', 'prosody']:
        raise ValueError("Invalid element, must be one of: 'phoneme' or 'prosody' ")

    # loop over subjects
    for subject in subjects:

        # save the model RDMs
        fname = DATADIR + subject + os.sep + 'behavior' + os.sep + 'rdm_models_' + element + '.txt'
        if os.path.exists(fname):
            print('%s - already exists' % fname)
        else:
            print('saving %s ...' % fname)

            # ---------- acoustic RDM ---------- #
            # import mfile to retrieve the original morph steps
            mfile = DATADIR + subject + os.sep + 'behavior' + os.sep + 'task_' + element + '.mat'
            mat = scipy.io.loadmat(mfile)
            query = [key for key in mat.keys() if '__' not in key][0]
            data = mat[query][0, :]  # [0,0]: male voice, [0,1]: female voice

            # compute acoustic RDM
            acoust_dist = list()
            for vi in range(len(data)):  # loop over voices
                levels = data[vi][0].flatten()  # five morph levels in a given voice
                dist = distance.pdist(levels[np.newaxis].T, metric='sqeuclidean')  # compute squared Euclidean dist.
                acoust_dist.append(dist / np.max(dist))  # normalize from 0-1
            acoust_dist = sum(acoust_dist) / len(acoust_dist)  # average over the voices
            AcoustRDM = distance.squareform(acoust_dist)  # vector form to a 5 x 5 matrix

            # ---------- categorical RDM ---------- #
            # import mfile to retrieve the expected proportion
            mfile = DATADIR + subject + os.sep + 'behavior' + os.sep + 'task_fit_' + element + '.mat'
            mat = scipy.io.loadmat(mfile)
            query = [key for key in mat.keys() if '__' not in key][0]
            data = mat[query][0, 0]

            # retrieve the fitted sigmoid parameters
            xs = data[0].flatten()
            a = data[3][0][0][0]  # x50
            b = data[3][0][0][1]  # slope

            # compute categorical RDM
            props = 1 / (1 + np.exp((a - xs) * b))  # based on the expected proportions after collapsing voices
            props = props.flatten()
            dist = distance.pdist(props[np.newaxis].T, metric='sqeuclidean')  # compute squared Euclidean dist.
            cat_dist = dist / np.max(dist)  # normalize from 0-1
            CatRDM = distance.squareform(cat_dist)  # vector form to a 5 x 5 matrix

            # the model RDMs into pandas DataFrame
            data = {'AcoustRDM': AcoustRDM.flatten(), 'CatRDM': CatRDM.flatten()}
            df = pd.DataFrame(data=data)
            df.to_csv(fname, sep='\t') # save the model RDMs