# Shared and distinct representational dynamics of phonemes and prosody in ventral and dorsal speech streams

This repository contains data and code accompaning:

Baek, S-C., Kim, S-G., Maess, B., Grigutsch, M., & Sammler, D., Shared and distinct representational dynamics of phonemes and prosody in ventral and dorsal speech streams.

## Environment

Except for the behavior data analysis, the code in this repository runs on a Python 3.10.5 environments.

For dependencies to run the code, you can find them `requirements.txt`.

To install the dependencies, run the following command:

```
pip install -r requirements.txt
```


To go around the clashes with the existing environment on your computer, we recommend you install the dependencies in a virtual environments.

If you use [Anaconda](https://www.anaconda.com/) or [miniconda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install), you can create the environment as below:

```
conda create --name <envname> python=3.10.5
conda activate <envname>
```


Or if you don't have Anaconda or miniconda installed, you can use Python virtual environments as below:

```
python3 -m vevn <envname>
source ./venv/Scripts/activate
```

In the case of using `venv`, we recommend Python>=3.8.x.

## Repository structure
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

`code/matlab` includes MATLAB code to analyze behavior data

`code/python` contains Python code applicable to neural data to replicate time-resolved representational similarity (RSA) and multivariate transfer (mTE) analayses based on regions of interest (ROIs).


`data` consists of individual subdirectories `sub-?? (01-29)`, where `behavior` and `MEG` data are stored in separate folders.

`behavior` contains raw behavior responses, sigmoidal fits, and model representational disimilarity matrices (RDMs).

`meg` contains neural RDMs, ROI-based time-resolved RSA and mTE analysis results.

In `group/meg`, group-level statistical information is stored.


`figs` has `jupyter-notebook` files to replicate Figs. 1-6 in the paper based on the data in `data`.


`stim` contains all the stimuli that were presented during the MEG experiment.

## Replication

To replicate the analyses implemented in the manuscript, first run the following MATLAB code in `code/matlab:

```
fit_linear_and_sigmoid.m
```

Then, run the following Python code in `code/python`:

```
analysis_pipeline.py
```

Please note that these codes do not create any files if there already exist results.

Then, you can delete the files in `sub-??/behavior`, `sub-??/meg`,  and `group/meg`.

However the files below should never be deleted!:

```
data/sub-??/behavior/task_phoneme.mat
data/sub-??/behavior/task_prosody.mat
data/sub-??/meg/rdm_rois_*.pickle
data/sub-??/meg/noise_cov_rois_*.pickle
```

In case you delete these data, you can download them again in this repository.



