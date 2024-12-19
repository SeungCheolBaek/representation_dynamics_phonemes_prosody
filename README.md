## Shared and distinct representational dynamics of phonemes and prosody in ventral and dorsal speech streams

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
* [data](./data)
	* [group](./data/group)
		* [meg](/data/group/meg)
	* [sub-??](./data/sub-01)
		* [behavior](./data/sub-01/behavior)
		* [meg](./data/sub-01/behaviour)
* [figs](./figs)
* [stim](./stim)
