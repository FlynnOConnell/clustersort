from __future__ import annotations

import numpy as np
import pywt
import umap
from scipy import linalg
from scipy.signal import find_peaks
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from statsmodels.stats.diagnostic import lilliefors

from clustersort.spk_config import SortConfig


def group_consecutives(arr):
    """
    Searches array for runs of consecutive numbers. Returns a list of lists
    of consecutive numbers found.

    Parameters
    ----------
    arr : list or np.array

    Returns
    -------
    list of lists
    """
    diff_arr = np.diff(arr)
    change = np.where(diff_arr>1)[0] + 1
    out = []
    prev = 0
    for i in change:
        out.append(arr[prev:i])
        prev = i

    out.append(arr[prev:])
    return out

def implement_wavelet_transform(waves, n_pc=10):
    coeffs = pywt.wavedec(waves, "haar", axis=1)
    all_coeffs = np.column_stack(coeffs)
    k_stats = np.zeros((all_coeffs.shape[1],))
    p_vals = np.ones((all_coeffs.shape[1],))
    for i, c in enumerate(all_coeffs.T):
        k_stats[i], p_vals[i] = lilliefors(c, dist="norm")

    idx = np.argsort(p_vals)
    return all_coeffs[:, idx[:n_pc]]

def implement_pca(scaled_slices):
    pca = PCA()
    pca_slices = pca.fit_transform(scaled_slices)
    return pca_slices, pca.explained_variance_ratio_

def implement_umap(waves, n_pc=3, n_neighbors=30, min_dist=0.0):
    reducer = umap.UMAP(n_components=n_pc, n_neighbors=n_neighbors, min_dist=min_dist)
    return reducer.fit_transform(waves)

def get_waveform_energy(waves):
    """
    Returns array of waveform energies

    Parameters
    ----------
    waves : np.array, matrix of waveforms, with row for each spike

    Returns
    -------
    np.array
    """
    return np.sqrt(np.sum(waves**2, axis=1)) / waves.shape[1]

def get_spike_slopes(waves):
    """
    Returns array of spike slopes (initial downward slope of spike)

    Parameters
    ----------
    waves : np.array, matrix of waveforms, with row for each spike

    Returns
    -------
    np.array
    """
    slopes = np.zeros((waves.shape[0],))
    for i, wave in enumerate(waves):
        peaks = find_peaks(wave)[0]
        minima = np.argmin(wave)
        if not any(peaks < minima):
            maxima = np.argmax(wave[:minima])
        else:
            maxima = max(peaks[np.where(peaks < minima)[0]])

        slopes[i] = (wave[minima] - wave[maxima]) / (minima - maxima)

    return slopes

def get_ISI_and_violations(spike_times, fs,):
    """
    Returns array of inter-spike-intervals (ms) and corresponding number of 1ms and 2ms violations.

    Parameters
    ----------
    spike_times : numpy.array
    fs : float, sampling rate in Hz

    Returns
    -------
    isi : numpy.array
        interspike-intervals for the given spike-times
    violations1 : int
        number of 1ms violations
    violations2 : int
        number of 2ms violations
    """
    fs = float(fs / 1000.0)
    isi = np.ediff1d(np.sort(spike_times)) / fs
    violations1 = np.sum(isi < 1.0)
    violations2 = np.sum(isi < 2.0)

    return isi, violations1, violations2

def scale_waveforms(waves, energy=None):
    """
    Scale the extracted spike waveforms by their energy.

    Parameters
    ----------
    waves : array-like
        The spike waveforms as a 2-D array.
    energy : array-like
        The energy of each spike waveform as a 1-D array.

    Returns
    -------
    scaled_slices : array-like
        The scaled spike waveforms as a 2-D array.
    """
    if energy is None:
        energy = get_waveform_energy(waves)
    elif len(energy) != waves.shape[0]:
        raise ValueError(
            (
                "Energies must correspond to each waveforms."
                "Different lengths are not allowed"
            )
        )

    scaled_slices = np.zeros(waves.shape)
    for i, w in enumerate(zip(waves, energy)):
        scaled_slices[i] = w[0] / w[1]

    return scaled_slices

def compute_waveform_metrics(waves, n_pc=3, use_umap=False):
    """

    Make clustering data array with columns:
         - amplitudes, energy, slope, pc1, pc2, pc3, etc.
    Parameters
    ----------
    waves : np.array
        waveforms with a row for each spike waveform
    n_pc : int (optional)
        number of principal components to include in data array
    use_umap : bool (optional)
        whether to use UMAP instead of PCA for dimensionality reduction

    Returns
    -------
    np.array
    """
    data = np.zeros((waves.shape[0], 3))
    for i, wave in enumerate(waves):
        data[i, 0] = np.min(wave)
        data[i, 1] = np.sqrt(np.sum(wave**2)) / len(wave)
        peaks = find_peaks(wave)[0]
        minima = np.argmin(wave)
        if not any(peaks < minima):
            maxima = np.argmax(wave[:minima])
        else:
            maxima = max(peaks[np.where(peaks < minima)[0]])

        data[i, 2] = (wave[minima] - wave[maxima]) / (minima - maxima)

    # Scale waveforms to energy before running PCA
    if use_umap:
        pc_waves = implement_umap(waves, n_pc=n_pc)
    else:
        scaled_waves = scale_waveforms(waves, energy=data[:, 1])
        pc_waves, _ = implement_pca(scaled_waves)

    data = np.hstack((data, pc_waves[:, :n_pc]))
    data_columns = ["amplitude", "energy", "spike_slope"]
    data_columns.extend(["PC%i" % i for i in range(n_pc)])
    return data, data_columns

def get_mahalanobis_distances_to_cluster(data, model, clusters, target_cluster):
    """
    Computes mahalanobis distance from spikes in target_cluster to all clusters in a GMM model

    Parameters
    ----------
    data : np.array,
        data used to train GMM
    model : GaussianMixture
        trained mixture model
    clusters : np.array
        maps data points to clusters
    target_cluster : int, cluster for which to compute distances

    Returns
    -------
    np.array
    """
    unique_clusters = np.unique(abs(clusters))
    out_distances = dict.fromkeys(unique_clusters)
    cluster_idx = np.where(clusters == target_cluster)[0]
    for other_cluster in unique_clusters:
        mahalanobis_dist = np.zeros((len(cluster_idx),))
        other_cluster_mean = model.means_[other_cluster, :]
        other_cluster_covar_I = linalg.inv(model.covariances_[other_cluster, :, :])
        for i, idx in enumerate(cluster_idx):
            mahalanobis_dist[i] = mahalanobis(
                data[idx, :], other_cluster_mean, other_cluster_covar_I
            )

        out_distances[other_cluster] = mahalanobis_dist

    return out_distances

def get_recording_cutoff(
    filt_el,
    sampling_rate,
    voltage_cutoff,
    max_breach_rate,
    max_secs_above_cutoff,
    max_mean_breach_rate_persec,
):
    """
    Determine the cutoff point for a recording based on the number of voltage violations.

    Parameters
    ----------
    filt_el : np.array
        Filtered electrophysiological data
    sampling_rate : int
        Sampling rate of the electrophysiological data
    voltage_cutoff : float
        Voltage cutoff for determining violations
    max_breach_rate : float
        Maximum allowed breach rate (breaches per second)
    max_secs_above_cutoff : int
        Maximum allowed number of seconds above the cutoff
    max_mean_breach_rate_persec : float
        Maximum allowed mean breach rate per second

    Returns
    -------
    int
        Recording cutoff in seconds

    """
    breach_idx = np.where(filt_el > voltage_cutoff)[0]
    breach_rate = float(len(breach_idx) * int(sampling_rate)) / len(filt_el)
    # truncate to nearest second and make 1 sec bins
    filt_el = filt_el[: int(sampling_rate) * int(len(filt_el) / sampling_rate)]
    test_el = np.reshape(filt_el, (-1, int(sampling_rate)))
    breaches_per_sec = [
        len(np.where(test_el[i] > voltage_cutoff)[0]) for i in range(len(test_el))
    ]
    breaches_per_sec = np.array(breaches_per_sec)
    secs_above_cutoff = len(np.where(breaches_per_sec > 0)[0])
    if secs_above_cutoff == 0:
        mean_breach_rate_persec = 0
    else:
        mean_breach_rate_persec = np.mean(
            breaches_per_sec[np.where(breaches_per_sec > 0)[0]]
        )

    # And if they all exceed the cutoffs, assume that the headstage fell off mid-experiment
    recording_cutoff = int(len(filt_el) / sampling_rate)  # cutoff in seconds
    if (
        breach_rate >= max_breach_rate
        and secs_above_cutoff >= max_secs_above_cutoff
        and mean_breach_rate_persec >= max_mean_breach_rate_persec
    ):
        # Find the first 1s epoch where the number of cutoff breaches is
        # higher than the maximum allowed mean breach rate
        recording_cutoff = np.where(breaches_per_sec > max_mean_breach_rate_persec)[0][
            0
        ]
        # cutoff is still in seconds since 1 sec bins

    return recording_cutoff

def UMAP_METRICS(waves, n_pc):
    return compute_waveform_metrics(waves, n_pc, use_umap=True)

class ClusterGMM:
    def __init__(
        self,
        config: SortConfig,
        n_iters: int = None,
        n_restarts: int = None,
        thresh: int | float = None,
    ):
        self.cluster_params = config.get_section("cluster")
        self.n_iters = n_iters if n_iters else int(self.cluster_params["max-iterations"])
        self.n_restarts = n_restarts if n_restarts else int(self.cluster_params["restarts"])
        self.thresh = thresh if thresh else float(self.cluster_params["convergence-criterion"])

    def fit(self, data_features, n_clusters):
        """
        Perform Gaussian Mixture Model (GMM) clustering on spike waveform data.

        This method takes pre-processed waveform data and applies GMM-based clustering
        to categorize each waveform into different neuronal units. It performs multiple
        checks for the validity of the clustering solution.

        Parameters
        ----------
        data_features : array_like
            A 2D array where each row represents a waveform and each column is a feature of the waveform
            (e.g., amplitude, principal component, etc.) down each column.
        n_clusters : int
            The number of clusters to use in the GMM.

        Returns
        -------
        best_model : GaussianMixture
            The best-fitting GMM object.
        predictions : array_like
            The cluster assignments for each data point as a 1-D array.
        min_bic : float
            The minimum Bayesian information criterion (BIC) value achieved across all restarts.

        """
        min_bic = None
        best_model = None

        for i in range(self.n_restarts):  # default: 10
            model = GaussianMixture(
                n_components=n_clusters,
                covariance_type="full",
                tol=self.thresh,  # default: 0.001
                random_state=i,
                max_iter=self.n_iters,  # default: 1000
            )
            model.fit(data_features)
            if model.converged_:
                new_bic = model.bic(data_features)
                if min_bic is None:
                    min_bic = model.bic(data_features)
                    best_model = model
                elif new_bic < min_bic:
                    best_model = model
                    min_bic = new_bic

        predictions = best_model.predict(data_features)
        self._model = best_model
        self._predictions = predictions
        self._bic = min_bic
        return best_model, predictions, min_bic
