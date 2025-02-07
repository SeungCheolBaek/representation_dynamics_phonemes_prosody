# Shared and distinct representational dynamics of phonemes and prosody in ventral and dorsal speech streams

This repository contains data and code accompaning:
Baek, S-C., Kim, S-G., Maess, B., Grigutsch, M., & Sammler, D., Shared and distinct representational dynamics of phonemes and prosody in ventral and dorsal speech streams. *bioRxiv*, 2025-01. [https://doi.org/10.1101/2025.01.24.634030](https://doi.org/10.1101/2025.01.24.634030).

This study investigates cortical representational dynamics of phonemes and prosody using time-resolved representational similarity (RSA) and multivariate transfer entropy (mTE) analyses applied to MEG and behavioral psychophysical data.

In this repository, we provide the auditory stimuli and individual behavioral responses collected during the MEG experiment, as well as the preprocessed neural data including the neural representational dissimilarity matrices (RDMs) and time-resolved RSA and mTE results based on the regions of interest (ROIs).

The scripts provided here are to replicate the ROI-based RSA and mTE analyses, as well as the main figures (Figs 1-6.).


# System Requirements
## Hardware requirements 
The scripts can run on a standard computer, but would run much faster if accessible to computing clusters.

## Software requirements
### OS requirements
The scripts is supported for *MacOS* and *Linux*, and has been tested on the following systems:
+ MacOS: Sonoma (14.7.1)
+ Linux: Debian GNU/Linux 11 (bullseye)

### MATLAB
The analysis of behavioral date relies on MATLAB (>= R2013a) with *Curve Fitting Toolbox*.
The MATLAB codes were developed using MATLAB R2021a.

### Python
The python codes were originally developed in a Python 3.10.5 environment, and has been tested with Python 3.9.13.
In general, we recommend Python>=3.8.x.

Python dependencies are listed below:

```
numpy
scipy
mne
joblib
tqdm
scikit-learn
pandas
matplotlib
seaborn
notebook
```


# Installation Guide
To run the scripts on your computer, first clone this repository as follows:

```
git clone https://github.com/SeungCheolBaek/representation_dynamics_phonemes_prosody.git
```

Next, run the following command in propmt to install all dependencies (<1 min. on a standard computer):

```
pip install -r requirements.txt
```

To avoid clashes with the existing environment on your computer, we recommend you install the dependencies in a virtual environment.

If you use [Anaconda](https://www.anaconda.com/) or [miniconda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install), you can create the environment as below:

```
conda create --name <envname> python=3.10.5
conda activate <envname>
```

Otherwise, you can use Python virtual environments as below:

```
python3 -m vevn <envname>
source venv/bin/activate
```


# Repository structure
* [code](./code)
	* [matlab](./code/matlab)
	* [python](./code/python)
* [data](./data)
	* [group](./data/group)
		* [meg](/data/group/meg)
	* [sub-??](./data/sub-01) (01-29)
		* [behavior](./data/sub-01/behavior)
		* [meg](./data/sub-01/behaviour)
* [figs](./figs)
* [stim](./stim)

`code` contains the codes to replicate the analyses implemented in the paper.

`code/matlab` includes MATLAB code to analyze the behavior data.

`code/python` stores the Python codes applicable to neural data to replicate ROI-based time-resolved RSA and mTE analysis.

`data` consists of individual sub-directories `sub-?? (01-29)`, where `behavior` and `meg` data are stored in separate folders.

`behavior` contains raw behavior responses, sigmoidal fits, and model RDMs.

`meg` contains neural RDMs, ROI-based time-resolved RSA and mTE analysis results.

In `group/meg`, group-level statistical information is stored.

`figs` has `jupyter-notebook` files to replicate Figs. 1-6 in the paper based on the data in `data`.

`stim` contains all the stimuli that were presented during the MEG experiment.


# Implementation

To replicate the analyses implemented in the manuscript, first run the following MATLAB code in `code/matlab`:

```
fit_linear_and_sigmoid.m
```

Then, run the following Python code in `code/python`:

```
analysis_pipeline.py
```

Please note that these commands do not create any files, if the results already exist in `data`.

To reproduce the results files, you can delete the files in `sub-??/behavior`, `sub-??/meg`,  and `group/meg`, and run the codes above again.
However, the files below should never be deleted:

```
data/sub-??/behavior/task_phoneme.mat
data/sub-??/behavior/task_prosody.mat
data/sub-??/meg/rdm_rois_*.pickle
data/sub-??/meg/noise_cov_rois_*.pickle
```
In case you delete these data, you can download them again in this repository.

On a standard computer, the whole computation would take about ~45-48 hours (tested with MacBook Pro, Apple M1, 16GB RAM).

