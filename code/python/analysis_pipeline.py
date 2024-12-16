## A pipeline for the ROI-based analyses
#
# written by S-C. Baek
# update: 16.12.2024
#
'''
The following code is to replicate the results of the ROI-based analyses in the paper.

To run this code, all python toolboxes in 'requirements.txt' under MAINDIR should be installed.
(Note: MAINDIR is a path to this repository stored on your computer or laptop ...)

Before run this code, 'fit_linear_and_sigmoid.m' under 'MAINDIR/code/matlab' directory should be run first.

This pipeline incorporates largely three main analyses including statistical inference:
    1) ROI-based RSA on task-agnostic data
    2) mTE analysis based on ROI-wise neural RDM time series
    3) ROI-based RSA on task-specific data (to investigate task-modulation effect).

The results to be created from each step will be stored in the following directory:
    'MAINDIR/data/sub-??/behavior/' (for individual model RDMs)
    'MAINDIR/data/sub-??/meg/' (for individual neural data)
    'MAINDIR/data/group/meg/' (for statistical inference at the group level).

The code to replicate the main figures (Figs. 1-6) based on the results can be found here: 'MAINDIR/figs/'.
'''
import os

from create_model_rdms import create_model_rdms
from run_rsa_rois import run_rsa_rois
from run_mte_rois import run_mte_rois, run_mte_rois_null, normalize_mte
from cluster_stats import cluster_1samp_ttest_rsa, cluster_1samp_ttest_mte, cluster_1samp_ttest_rsa_task_modul
from utils import get_main_dir, sort_rois_subset

# path settings
MAINDIR = get_main_dir()
DATADIR = MAINDIR + os.sep + 'data/'

# subjects
s = os.listdir(DATADIR)
subjects = [i for i in s if 'sub-' in i]
subjects.sort()

# linguistic elements
elements = ['phoneme', 'prosody']

# loop over elements
for element in elements:

    # ---------- 1) RSA on task-agnostic data ---------- #
    # Step 01 - create model RDMs
    create_model_rdms(subjects, element)
    # Step 02 - run ROI-based time-resolved RSA to investigate general representations of phonemes or prosody
    run_rsa_rois(subjects, element, tbeg=-0.2, twindow=0.024, gen=True,
                 task='', white=True, n_jobs=-1, verbose=True)
    # Step 03 - group-level statistics for general representations
    cluster_1samp_ttest_rsa(subjects, element, testvar='gen', n_permutations=100000,
                            p_threshold=0.05, tail=1, n_jobs=-1, verbose=True)
    # Step 04 - run ROI-based time-resolved RSA to investigate unique representations of phonemes or prosody
    run_rsa_rois(subjects, element, tbeg=-0.2, twindow=0.024, gen=False,
                 task='', white=True, n_jobs=-1, verbose=True)
    # Step 05 - group-level statistics for unique representations
    cluster_1samp_ttest_rsa(subjects, element, testvar='uni', n_permutations=100000,
                            p_threshold=0.05, tail=1, n_jobs=-1, verbose=True)

    # ---------- 2) mTE analysis on ROI-wise (subset) neural RDMs ---------- #
    # Step 06 - sort out a subset of ROIs for each element that showed significant corresponding representations
    rois_subset_indices = sort_rois_subset(element)
    # Step 07 - compute raw mTE for pairs of ROIs
    run_mte_rois(subjects, element, rois_subset_indices=rois_subset_indices,
                 max_lag=0.3, white=True, n_jobs=-1, verbose=False)
    # Step 08 - compute null mTE for pairs of ROIs using permutation
    run_mte_rois_null(subjects, element, rois_subset_indices=rois_subset_indices,
                      max_lag=0.3, white=True, n_perms=200, n_jobs=-1, verbose=False)
    # Step 09 - normalize raw mTE using null distribution
    normalize_mte(subjects, element)
    # Step 10 - group-level statistics for mTEn
    cluster_1samp_ttest_mte(subjects, element, n_permutations=100000, p_threshold=0.05, tail=1, n_jobs=-1, verbose=True)

    # ---------- 3) RSA on task-specific data & testing task-modulation effect ---------- #
    # Step 11 - run ROI-based time-resolved RSA to investigate general representations of phonemes or prosody
    #           during the specific task
    run_rsa_rois(subjects, element, tbeg=-0.2, twindow=0.024, gen=True,
                 task='att', white=True, n_jobs=-1, verbose=True) # during the relevant task
    run_rsa_rois(subjects, element, tbeg=-0.2, twindow=0.024, gen=True,
                 task='ign', white=True, n_jobs=-1, verbose=True) # during the irrelevant task
    # Step 12 - group-level statistics for task-modulation effect on general representations
    cluster_1samp_ttest_rsa_task_modul(subjects, element, testvar='gen', rois_subset_indices=rois_subset_indices,
                                       n_permutations=100000, p_threshold=0.05, tail=1, n_jobs=-1, verbose=True)
    # Step 13 - run ROI-based time-resolved RSA to investigate unique representations of phonemes or prosody
    #           during the specific task
    run_rsa_rois(subjects, element, tbeg=-0.2, twindow=0.024, gen=False,
                 task='att', white=True, n_jobs=-1, verbose=True)  # during the relevant task
    run_rsa_rois(subjects, element, tbeg=-0.2, twindow=0.024, gen=False,
                 task='ign', white=True, n_jobs=-1, verbose=True)  # during the irrelevant task
    # Step 14 - sort out a subset from a subset of ROIs that showed significant task-modulation effect
    #           on the general representations of phonemes
    roi_subset_task_indices = sort_rois_subset(element, between_conditions=True)
    # Step 15 - group-level statistics for task-modulation effect on unique representations
    if len(roi_subset_task_indices) > 0: # only when significant effect on general representations
        cluster_1samp_ttest_rsa_task_modul(subjects, element, testvar='uni',
                                           rois_subset_indices=rois_subset_indices[roi_subset_task_indices],
                                           n_permutations=100000, p_threshold=0.05, tail=1, n_jobs=-1, verbose=True)