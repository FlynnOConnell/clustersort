"""
========
Autosort
========
.. currentmodule:: spk2py.autosort

Autosort: A Python package for spike sorting.

This package is designed to automate the spike sorting process for extracellular recordings.

It is designed to be used with the Spike2 software package from CED, with pl2 files from Plexon
in development.

The package is designed to be used with the AutoSort pipeline, which is a series of steps that
are performed on the data to extract spikes and cluster them. The pipeline is as follows:

1. Read in the data from the pl2 file.
2. Filter the data.
3. Extract spikes from the filtered data.
4. Cluster the spikes.
5. Perform breach analysis on the clusters.
6. Resort the clusters based on the breach analysis.
7. Save the data to an HDF5 file, and graphs to given plotting folders.

"""
import numpy as np
from .autosort import *
from .directory_manager import *
from .spk_config import *
from .wf_shader import *


def cluster_gmm(data, n_clusters, n_iter, restarts, threshold):
    """
    Cluster the data using a Gaussian Mixture Model (GMM).

    Parameters
    ----------
    data : array-like
        The input data to be clustered, generally the output of a PCA, as a 2-D array.
    n_clusters : int
        The number of clusters to use in the GMM.
    n_iter : int
        The maximum number of iterations to perform in the GMM.
    restarts : int
        The number of times to restart the GMM with different initializations.
    threshold : float
        The convergence threshold for the GMM.

    Returns
    -------
    best_fit_gmm : object
        The best-fitting GMM object.
    predictions : array-like
        The cluster assignments for each data point as a 1-D array.
    min_bayesian : float
        The minimum Bayesian information criterion (BIC) value achieved across all restarts.
    """

    g = []
    bayesian = []

    # Run the GMM
    try:
        for i in range(restarts):
            g.append(
                GaussianMixture(
                    n_components=n_clusters,
                    covariance_type="full",
                    tol=threshold,
                    random_state=i,
                    max_iter=n_iter,
                )
            )
            g[-1].fit(data)
            if g[-1].converged_:
                bayesian.append(g[-1].bic(data))
            else:
                del g[-1]

        # print len(akaike)
        bayesian = np.array(bayesian)
        best_fit = np.where(bayesian == np.min(bayesian))[0][0]

        predictions = g[best_fit].predict(data)
        return g[best_fit], predictions, np.min(bayesian)
    except Exception as e:
        logger.warning(f"Error in clusterGMM: {e}", exc_info=True)


def get_lratios(data, predictions):
    """
    Calculate L-ratios for each cluster, a measure of cluster quality.

    Parameters
    ----------
    data : array-like
        The input data, generally the output of a PCA, as a 2-D array.
    predictions : array-like
        The cluster assignments for each data point as a 1-D array.

    Returns
    -------
    Lrats : dict
        A dictionary with cluster labels as keys and the corresponding L-ratio values as values.
    """
    Lrats = {}
    df = np.shape(data)[1]
    for ref_cluster in np.sort(np.unique(predictions)):
        if ref_cluster < 0:
            continue
        ref_mean = np.mean(data[np.where(predictions == ref_cluster)], axis=0)
        ref_covar_I = linalg.inv(
            np.cov(data[np.where(predictions == ref_cluster)], rowvar=False)
        )
        Ls = [
            1 - chi2.cdf((mahalanobis(data[point, :], ref_mean, ref_covar_I)) ** 2, df)
            for point in np.where(predictions[:] != ref_cluster)[0]
        ]
        Lratio = sum(Ls) / len(np.where(predictions == ref_cluster)[0])
        Lrats[ref_cluster] = Lratio
    return Lrats



def scale_waveforms(slices_dejittered):
    """
    Scale the extracted spike waveforms by their energy.

    Parameters
    ----------
    slices_dejittered : array-like
        The dejittered spike waveforms as a 2-D array.

    Returns
    -------
    scaled_slices : array-like
        The scaled spike waveforms as a 2-D array.
    energy : array-like
        The energy of each spike waveform as a 1-D array.
    """
    energy = np.sqrt(np.sum(slices_dejittered ** 2, axis=1)) / len(slices_dejittered[0])
    scaled_slices = np.zeros((len(slices_dejittered), len(slices_dejittered[0])))
    for i in range(len(slices_dejittered)):
        scaled_slices[i] = slices_dejittered[i] / energy[i]

    return scaled_slices, energy
