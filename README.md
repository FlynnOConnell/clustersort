# clustersort

Data analysis pipeline for electrophysiological data.

Heavily inspired by [blechpy](https://github.com/nubs01/blechpy)

[![Documentation](https://img.shields.io/badge/view-Documentation-blue?style=)](https://flynnoconnell.github.io/clustersort/index.html# "Go to project documentation")
[![Documentation Status](https://readthedocs.org/projects/clustersort/badge/?version=latest)](https://clustersort.readthedocs.io/en/latest/?badge=latest)
![CircleCI](https://dl.circleci.com/status-badge/img/gh/FlynnOConnell/clustersort/tree/master.svg?style=shield)

### Default File structure

```
~/
├── autosort
│   ├── h5
│   ├── completed
│   ├── results
│   ├── to_run
│   └── autosort_config.ini
```

## Configuration Guide

### `default_config` Function

The `default_config` function initializes a configuration file with default settings and creates necessary directories for the run. It accepts the following parameters:

- **`path: Path`**: The path where the configuration file will be saved.
- **`config_ver: int`**: The version of the configuration to be used. The default value is 5.

## Configuration Sections and Parameters

### `run-settings`

Parameters defining the software and hardware configurations:

- **`resort-limit`**: The maximum number of times the sorting process can be rerun. Default is '3'.
- **`cores-used`**: The number of cores to be used during the run. Default is '8'.
- **`weekday-run`**: The number of runs allowed on a weekday. Default is '2'.
- **`weekend-run`**: The number of runs allowed on a weekend. Default is '8'.
- **`run-type`**: Defines the type of run (Auto/Manual). Default is 'Auto'.
- **`manual-run`**: The number of manual runs allowed. Default is '2'.

### `paths`

Here we define various paths necessary for the script, set by default to subdirectories in the parent directory of the specified `path`:

- **`run-path`**: Path to the directory where files to be processed are stored.
- **`results-path`**: Path to the directory where results will be stored.
- **`completed-path`**: Path where completed files will be moved.

### `clustering`

Parameters defining the clustering process:

- **`max-clusters`**: Maximum number of clusters to use in the clustering algorithm. Default is '7'.
- **`max-iterations`**: Maximum number of iterations for the clustering algorithm. Default is '1000'.
- **`convergence-criterion`**: The criterion for convergence in the clustering algorithm. Default is '.0001'.
- **`random-restarts`**: Number of random restarts in the clustering process to avoid local minima. Default is '10'.
- **`l-ratio-cutoff`**: The cutoff value for the L-Ratio metric, used to assess cluster quality. Default is '.1'.

### `signal`

Parameters involved in signal preprocessing and spike detection:

- **`disconnect-voltage`**: Voltage level that indicates a disconnection in the signal, to detect noise or artifacts. Default is '1500'.
- **`max-breach-rate`**: The maximum rate at which breaches (potentially signal artifacts or spikes) can occur before it is considered noise. Default is '.2'.
- **`max-breach-count`**: The maximum count of breaches allowed in a given window of time. Default is '10'.
- **`max-breach-avg`**: Perhaps the average breach level over a defined window. Default is '20'.
- **`intra-hpc_cluster-cutoff`**: A cutoff value for considering a signal as noise based on some intra-cluster metric. Default is '3'.

### `filtering`

Filtering parameters to isolate the frequency range of interest:

- **`low-cutoff`**: The low cutoff frequency for a band-pass filter. Default is '600'.
- **`high-cutoff`**: The high cutoff frequency for the band-pass filter. Default is '3000'.

### `spike`

Spike detection and extraction parameters:

- **`pre-time`**: Time before a spike event to include in each spike waveform, in seconds. Default is '.2'.
- **`post-time`**: Time after a spike event to include in each spike waveform, in seconds. Default is '.6'.
- **`sampling-rate`**: The sampling rate of the recording, in Hz. Default is '20000'.

### `std-dev`

Standard deviation parameters for spike detection and artifact removal:

- **`spike-detection`**: A multiplier for the standard deviation of the noise to set a threshold for spike detection. Default is '2.0'.
- **`artifact-removal`**: A threshold for artifact removal, based on a multiple of the standard deviation. Default is '10.0'.

### `pca`

Parameters defining how principal component analysis (PCA) is conducted on the spike waveforms:

- **`variance-explained`**: The proportion of variance explained to determine the number of principal components to retain. Default is '.95'.
- **`use-percent-variance`**: Whether to use percent variance to determine the number of components to retain. Default is '1'.
- **`principal-component-n`**: An alternative to variance-explained, specifying the number of principal components to retain directly. Default is '5'.

### `post-process`

Post-processing parameters:

- **`reanalyze`**: Whether to reanalyze the data. Default is '0'.
- **`simple-gmm`**: Whether to use a simple Gaussian Mixture Model in the post-processing. Default is '1'.
- **`image-size`**: The size of images generated during post-processing. Default is '70'.
- **`temporary-dir`**: The directory to store temporary files during processing. Default is the user's home directory followed by '/tmp_python'.

### `version`

Version control parameters:

- **`config-version`**: The version of the configuration. Default is determined by the `config_ver` parameter passed to the `default_config` function.

### Usage

To initialize a configuration file with default settings, use the `default_config` function as follows:

```python
from pathlib import Path
from clustersort.spk_config import default_config

default_config(path=Path('/path/to/your/config.ini'))
```
---

