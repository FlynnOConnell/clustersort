# -*- coding: utf-8 -*-
from __future__ import annotations

import configparser
from collections import namedtuple
import logging
import os
import shutil
import warnings
from datetime import date
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ImageFont, ImageDraw, Image
from matplotlib import cm
from scipy import linalg
from scipy.spatial.distance import mahalanobis

from directory_manager import DirectoryManager
from spk2py.autosort.utils.wf_shader import waveforms_datashader
from spk2py.cluster import clusterGMM, get_Lratios, scale_waveforms, implement_pca

logger = logging.getLogger(__name__)
logpath = Path().home() / "autosort" / "directory_logs.log"
logging.basicConfig(filename=logpath, level=logging.DEBUG)
logger.addHandler(logging.StreamHandler())


# Factory
def run_spk_process(filename, data, params, dir_manager, chan_num):
    """
    Data can be in several formats:
    1) Dictionary with keys 'spikes' and 'times'
    2) Named tuple with fields 'spikes' and 'times'
    # TODO: Add support for pandas DataFrame and other input methods
    """
    input_data = {}
    if isinstance(data, dict):
        input_data["spikes"] = data["spikes"]
        input_data["times"] = data["times"]
    else:
        raise TypeError("Data must be a dictionary with keys 'spikes' and 'times'")

    proc = ProcessChannel(filename, input_data, params, dir_manager, chan_num)
    proc.process_channel()


def infofile(filename, path, sort_time, params):
    # dumps run info to a .info file
    config = configparser.ConfigParser()
    config["METADATA"] = {
        "h5 File": filename,
        "Run Time": sort_time,
        "Run Date": date.today().strftime("%m/%d/%y"),
    }
    config["PARAMS USED"] = params
    with open(
        path + "/" + os.path.splitext(filename)[0] + "_" + "sort.info", "w"
    ) as info_file:
        config.write(info_file)


class ProcessChannel:
    def __init__(self, filename, data, params, dir_manager, chan_num):
        self.filename = filename
        self.data = data
        # self.metadata = metadata
        self.params = params
        self.sampling_rate = 18518.52  # TODO: Get this from metadata
        self.chan_num = chan_num
        self.dir_manager = dir_manager

    @property
    def pvar(self):
        return float(self.params.pca["variance-explained"])

    @property
    def usepvar(self):
        return int(self.params.pca["use-percent-variance"])

    @property
    def userpc(self):
        return int(self.params.pca["principal-component-n"])

    @property
    def max_clusters(self):
        return int(self.params.cluster["max-clusters"])

    @property
    def max_iterations(self):
        return int(self.params.cluster["max-iterations"])

    @property
    def thresh(self):
        return float(self.params.cluster["convergence-criterion"])

    @property
    def num_restarts(self):
        return int(self.params.cluster["random-restarts"])

    @property
    def wf_amplitude_sd_cutoff(self):
        return float(self.params.cluster["intra-cluster-cutoff"])

    @property
    def artifact_removal(self):
        return float(self.params.cluster["artifact-removal"])

    @property
    def pre_time(self):
        return float(self.params.spike["pre_time"])

    @property
    def post_time(self):
        return float(self.params.spike["post_time"])

    @property
    def bandpass(self):
        return float(self.params.filter["low-cutoff"]), float(
            self.params.filter["high-cutoff"]
        )

    @property
    def spike_detection(self):
        return int(self.params["spike-detection"])

    @property
    def STD(self):
        return int(self.params.detection["spike-detection"])

    @property
    def max_breach_rate(self):
        return float(self.params.breach["max-breach-rate"])

    @property
    def max_breach_count(self):
        return float(self.params.breach["max-breach-count"])

    @property
    def max_breach_avg(self):
        return float(self.params.breach["max-breach-avg"])

    @property
    def voltage_cutoff(self):
        return float(self.params.breach["voltage-cutoff"])

    def process_channel(
        self,
    ):
        while True:
            if self.data["spikes"].size == 0:
                (
                    self.dir_manager.reports
                    / f"channel_{self.chan_num + 1}"
                    / "no_spikes.txt"
                ).write_text(
                    "No spikes were found on this channel."
                    " The most likely cause is an early recording cutoff."
                )
                warnings.warn(
                    "No spikes were found on this channel. The most likely cause is an early recording cutoff."
                )
                (
                    self.dir_manager.reports
                    / f"channel_{self.chan_num + 1}"
                    / "success.txt"
                ).write_text("Sorting finished. No spikes found")
                return

            # Dejitter these spike waveforms, and get their maximum amplitudes
            amplitudes = np.min(self.data["spikes"], axis=1)

            np.save(
                self.dir_manager.intermediate
                / f"channel_{self.chan_num + 1}"
                / "spike_waveforms.npy",
                self.data["spikes"],
            )
            np.save(
                self.dir_manager.intermediate
                / f"channel_{self.chan_num + 1}"
                / "spike_times.npy",
                self.data["times"],
            )

            # Scale the dejittered spikes by the energy of the waveforms and perform PCA
            scaled_slices, energy = scale_waveforms(self.data["spikes"])
            pca_slices, explained_variance_ratio = implement_pca(scaled_slices)
            cumulvar = np.cumsum(explained_variance_ratio)
            graphvar = list(cumulvar[0 : np.where(cumulvar > 0.999)[0][0] + 1])

            if self.usepvar == 1:
                n_pc = np.where(cumulvar > self.pvar)[0][0] + 1
            else:
                n_pc = self.userpc

            np.save(
                self.dir_manager.intermediate
                / f"channel_{self.chan_num + 1}"
                / "spike_waveforms.npy",
                scaled_slices,
            )
            np.save(
                self.dir_manager.intermediate
                / f"channel_{self.chan_num + 1}"
                / "spike_times.npy",
                self.data["times"],
            )
            np.save(
                self.dir_manager.intermediate
                / f"channel_{self.chan_num + 1}"
                / "spike_waveforms_pca.npy",
                pca_slices,
            )

            # explained variance
            var = float(
                cumulvar[n_pc - 1]
            )  # mainly to avoid type checking issues in the annotation below
            fig = plt.figure()
            x = np.arange(0, len(graphvar) + 1)
            graphvar.insert(0, 0)
            plt.plot(x, graphvar)
            plt.vlines(n_pc, 0, 1, colors="r")
            plt.annotate(
                str(n_pc)
                + " PC's used for GMM.\nVariance explained= "
                + str(round(var, 3))
                + "%.",
                (n_pc + 0.25, cumulvar[n_pc - 1] - 0.1),
            )
            plt.title("Variance ratios explained by PCs (cumulative)")
            plt.xlabel("PC #")
            plt.ylabel("Explained variance ratio")
            fig.savefig(
                self.dir_manager.plots
                / f"channel_{self.chan_num + 1}"
                / "pca_variance.png",
                bbox_inches="tight",
            )
            plt.close("all")

            # Make an array of the data to be used for clustering
            data = np.zeros((len(pca_slices), n_pc + 2))
            data[:, 2:] = pca_slices[:, :n_pc]
            data[:, 0] = energy[:] / np.max(energy)
            data[:, 1] = np.abs(amplitudes) / np.max(np.abs(amplitudes))
            self.spk_gmm(data, self.data["times"], n_pc, amplitudes)
            break

    def spk_gmm(self, data, times_final, n_pc, amplitudes):
        for i in range(self.max_clusters - 2):
            try:
                model, predictions, bic = clusterGMM(
                    data,
                    n_clusters=i + 3,
                    n_iter=self.max_iterations,
                    restarts=self.num_restarts,
                    threshold=self.thresh,
                )
            except Exception as e:
                logger.warning("Error in clusterGMM", exc_info=True)
                continue

            if np.any(
                [
                    len(np.where(predictions[:] == cluster)[0]) <= n_pc + 2
                    for cluster in range(i + 3)
                ]
            ):
                plots_waveforms_ISIs_path = (
                    self.dir_manager.plots
                    / f"channel_{self.chan_num + 1}/{i + 3}_clusters_waveforms_ISIs"
                )
                plots_waveforms_ISIs_path.mkdir(parents=True, exist_ok=True)

                # Create and write to the invalid_sort.txt files
                with open(
                    self.dir_manager.plots
                    / f"channel_{self.chan_num + 1}"
                    / "invalid_sort.txt",
                    "w+",
                ) as f:
                    f.write(
                        "There are too few waveforms to properly sort this clustering"
                    )

                with open(
                    self.dir_manager.plots
                    / f"channel_{self.chan_num + 1}"
                    / "invalid_sort.txt",
                    "w+",
                ) as f:
                    f.write(
                        "There are too few waveforms to properly sort this clustering"
                    )
                continue
            # Sometimes large amplitude noise waveforms hpc_cluster with the spike waveforms because the amplitude has
            # been factored out of the scaled spikes. Run through the clusters and find the waveforms that are more than
            # wf_amplitude_sd_cutoff larger than the hpc_cluster mean.
            for cluster in range(i + 3):
                cluster_points = np.where(predictions[:] == cluster)[0]
                this_cluster = predictions[cluster_points]
                cluster_amplitudes = amplitudes[cluster_points]
                cluster_amplitude_mean = np.mean(cluster_amplitudes)
                cluster_amplitude_sd = np.std(cluster_amplitudes)
                reject_wf = np.where(
                    cluster_amplitudes
                    <= cluster_amplitude_mean
                    - self.wf_amplitude_sd_cutoff * cluster_amplitude_sd
                )[0]
                this_cluster[reject_wf] = -1
                predictions[cluster_points] = this_cluster

                # Make folder for results of i+2 clusters, and store results there
                clusters_path = (
                    self.dir_manager.plots
                    / f"clustering_results/channel_{self.chan_num + 1}/clusters{i + 3}"
                )
                clusters_path.mkdir(parents=True, exist_ok=True)

                np.save(clusters_path / "predictions.npy", predictions)
                np.save(clusters_path / "bic.npy", bic)

                # Plot the graphs, for this set of clusters, in the directory made for this channel
                plots_path = (
                    self.dir_manager.plots
                    / f"channel_{self.chan_num + 1}/{i + 3}_clusters"
                )
                plots_path.mkdir(parents=True, exist_ok=True)

            # Ignore cm.rainbow type checking because the dynamic __init__.py isn't recognized
            # noinspection PyUnresolvedReferences PyTypeChecker
            colors = cm.rainbow(np.linspace(0, 1, i + 3))
            for feature1 in range(len(data[0])):
                for feature2 in range(len(data[0])):
                    if feature1 < feature2:
                        fig = plt.figure()
                        plt_names = []
                        for cluster in range(i + 3):
                            plot_data = np.where(predictions[:] == cluster)[0]
                            plt_names.append(
                                plt.scatter(
                                    data[plot_data, feature1],
                                    data[plot_data, feature2],
                                    color=colors[cluster],
                                    s=0.8,
                                )
                            )

                        plt.xlabel("Feature %i" % feature1)
                        plt.ylabel("Feature %i" % feature2)
                        # Produce figure legend
                        plt.legend(
                            tuple(plt_names),
                            tuple("Cluster %i" % cluster for cluster in range(i + 3)),
                            scatterpoints=1,
                            loc="lower left",
                            ncol=3,
                            fontsize=8,
                        )
                        plt.title("%i clusters" % (i + 3))
                        fig.savefig(
                            self.dir_manager.plots
                            / f"channel_{self.chan_num + 1}/{i + 3}_clusters/feature{feature2}vs{feature1}.png",
                        )
                        plt.close("all")

            for ref_cluster in range(i + 3):
                fig = plt.figure()
                ref_mean = np.mean(data[np.where(predictions == ref_cluster)], axis=0)
                ref_covar_I = linalg.inv(
                    np.cov(data[np.where(predictions == ref_cluster)[0]], rowvar=False)
                )
                xsave = None
                for other_cluster in range(i + 3):
                    mahalanobis_dist = [
                        mahalanobis(data[point, :], ref_mean, ref_covar_I)
                        for point in np.where(predictions[:] == other_cluster)[0]
                    ]
                    # Plot histogram of Mahalanobis distances
                    y, binEdges = np.histogram(mahalanobis_dist, bins=25)
                    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
                    plt.plot(
                        bincenters, y, label="Dist from hpc_cluster %i" % other_cluster
                    )
                    if other_cluster == ref_cluster:
                        xsave = bincenters

                plt.xlim([0, max(xsave) + 5])
                plt.xlabel("Mahalanobis distance")
                plt.ylabel("Frequency")
                plt.legend(loc="upper right", fontsize=8)
                plt.title(
                    "Mahalanobis distance of all clusters from Reference Cluster: %i"
                    % ref_cluster
                )
                fig.savefig(
                    self.dir_manager.plots
                    / f"channel_{self.chan_num + 1}/{i + 3}_clusters/Mahalonobis_cluster{ref_cluster}.png",
                )
                plt.close("all")

            # Create file, and plot spike waveforms for the different clusters. Plot 10 times downsampled
            # dejittered/smoothed waveforms. Plot the ISI distribution of each hpc_cluster
            for cluster in range(i + 3):
                clust_path = (
                    self.dir_manager.plots
                    / f"channel_{self.chan_num + 1}/{i + 3}_clusters_waveforms_ISIs"
                )
                clust_path.mkdir(parents=True, exist_ok=True)

            x = np.arange(len(self.data["spikes"][0]) / 10) + 1
            ISIList = []
            for cluster in range(i + 3):
                cluster_points = np.where(predictions[:] == cluster)[0]
                fig, ax = plt.subplots()
                fig, ax = waveforms_datashader(
                    self.data["spikes"][cluster_points, :],
                    x,
                    self.dir_manager.filename
                    + "_datashader_temp_el"
                    + str(self.chan_num + 1),
                )
                ax.set_xlabel(
                    "Sample ({:d} samples per ms)".format(
                        int(self.sampling_rate / 1000)
                    )
                )
                ax.set_ylabel("Voltage (microvolts)")
                ax.set_title("Cluster%i" % cluster)
                fig.savefig(
                    self.dir_manager.plots
                    / f"channel_{self.chan_num + 1}/{i + 3}_clusters_waveforms_ISIs/Cluster{cluster}_waveforms"
                )
                plt.close("all")

                fig = plt.figure()
                cluster_times = times_final[cluster_points]
                ISIs = np.ediff1d(np.sort(cluster_times))
                ISIs = ISIs / 40.0
                plt.hist(
                    ISIs,
                    bins=[
                        0.0,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        8.0,
                        9.0,
                        10.0,
                        np.max(ISIs),
                    ],
                )
                plt.xlim([0.0, 10.0])
                plt.title(
                    "2ms ISI violations = %.1f percent (%i/%i)"
                    % (
                        (
                            float(len(np.where(ISIs < 2.0)[0]))
                            / float(len(cluster_times))
                        )
                        * 100.0,
                        len(np.where(ISIs < 2.0)[0]),
                        len(cluster_times),
                    )
                    + "\n"
                    + "1ms ISI violations = %.1f percent (%i/%i)"
                    % (
                        (
                            float(len(np.where(ISIs < 1.0)[0]))
                            / float(len(cluster_times))
                        )
                        * 100.0,
                        len(np.where(ISIs < 1.0)[0]),
                        len(cluster_times),
                    )
                )
                fig.savefig(
                    self.dir_manager.plots
                    / f"channel_{self.chan_num + 1}/{i + 3}_clusters_waveforms_ISIs/Cluster{cluster}_ISIs"
                )
                plt.close("all")
                ISIList.append(
                    "%.1f"
                    % (
                        (
                            float(len(np.where(ISIs < 1.0)[0]))
                            / float(len(cluster_times))
                        )
                        * 100.0
                    )
                )

            # Get isolation statistics for each solution
            Lrats = get_Lratios(data, predictions)

            isodf = pd.DataFrame(
                {
                    "IsoRating": "TBD",
                    "File": self.dir_manager.filename,
                    "Channel": self.chan_num + 1,
                    "Solution": i + 3,
                    "Cluster": range(i + 3),
                    "wf count": [
                        len(np.where(predictions[:] == cluster)[0])
                        for cluster in range(i + 3)
                    ],
                    "ISIs (%)": ISIList,
                    "L-Ratio": [round(Lrats[cl], 3) for cl in range(i + 3)],
                }
            )
            cluster_path = (
                self.dir_manager.reports
                / f"channel_{self.chan_num + 1}"
                / f"clusters_{i + 3}"
            )
            cluster_path.mkdir(parents=True, exist_ok=True)
            isodf.to_csv(
                cluster_path / "isoinfo.csv",
                index=False,
            )
            # output this all in a plot in the plots folder and replace the ISI plot in superplots
            for cluster in range(i + 3):
                text = (
                    "wf count: \n1 ms ISIs: \nL-Ratio: "  # package text to be plotted
                )
                text2 = "{}\n{}%\n{}".format(
                    isodf["wf count"][cluster],
                    isodf["ISIs (%)"][cluster],
                    isodf["L-Ratio"][cluster],
                )
                blank = (
                    np.ones((480, 640, 3), np.uint8) * 255
                )  # initialize empty white image

                cv2_im_rgb = cv2.cvtColor(
                    blank, cv2.COLOR_BGR2RGB
                )  # convert to color space pillow can use
                pil_im = Image.fromarray(cv2_im_rgb)  # get pillow image
                draw = ImageDraw.Draw(pil_im)  # create draw object for text
                draw.multiline_text(
                    (90, 100),
                    text,
                    fill=(0, 0, 0, 255),
                    spacing=50,
                    align="left",
                )  # draw the text
                draw.multiline_text(
                    (380, 100), text2, fill=(0, 0, 0, 255), spacing=50
                )  # draw the text
                isoimg = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
                temp_filename = str(
                    self.dir_manager.plots
                    / f"channel_{self.chan_num + 1}"
                    / f"clusters_{i + 3}"
                    / f"cluser_{cluster}_isoimg.png"
                )
                cv2.imwrite(
                    temp_filename,
                    isoimg,
                )  # save the image
        with open(
            self.dir_manager.reports / f"channel_{self.chan_num + 1}" / "success.txt",
            "w+",
        ) as f:
            f.write("Congratulations, this channel was sorted successfully")

    def superplots(self, maxclust):
        path = self.dir_manager.plots / f"channel_{self.chan_num + 1}"
        outpath = self.dir_manager.plots / f"channel_{self.chan_num + 1}" / "superplots"
        if outpath.exists():
            shutil.rmtree(outpath)
        outpath.mkdir(parents=True, exist_ok=True)
        for channel in outpath.glob("*"):
            try:
                currentpath = path + "/" + channel
                os.mkdir(
                    outpath + "/" + channel
                )  # create an output path for each channel
                for soln in range(
                    3, maxclust + 1
                ):  # for each number hpc_cluster solution
                    finalpath = outpath + "/" + channel + "/" + str(soln) + "_clusters"
                    os.mkdir(finalpath)  # create output folders
                    for cluster in range(0, soln):  # for each hpc_cluster
                        mah = cv2.imread(
                            currentpath
                            + "/"
                            + str(soln)
                            + "_clusters/Mahalonobis_cluster"
                            + str(cluster)
                            + ".png"
                        )
                        if not np.shape(mah)[0:2] == (480, 640):
                            mah = cv2.resize(mah, (640, 480))
                        wf = cv2.imread(
                            currentpath
                            + "/"
                            + str(soln)
                            + "_clusters_waveforms_ISIs/Cluster"
                            + str(cluster)
                            + "_waveforms.png"
                        )
                        if not np.shape(mah)[0:2] == (1200, 1600):
                            wf = cv2.resize(wf, (1600, 1200))
                        isi = cv2.imread(
                            currentpath
                            + "/"
                            + str(soln)
                            + "_clusters_waveforms_ISIs/Cluster"
                            + str(cluster)
                            + "_Isolation.png"
                        )
                        if not np.shape(isi)[0:2] == (480, 640):
                            isi = cv2.resize(isi, (640, 480))
                        blank = (
                            np.ones((240, 640, 3), np.uint8) * 255
                        )  # make whitespace for info
                        text = (
                            "Electrode: "
                            + channel
                            + "\nSolution: "
                            + str(soln)
                            + "\nCluster: "
                            + str(cluster)
                        )  # text to output to whitespace (hpc_cluster, channel, and solution numbers)
                        cv2_im_rgb = cv2.cvtColor(
                            blank, cv2.COLOR_BGR2RGB
                        )  # convert to color space pillow can use
                        pil_im = Image.fromarray(cv2_im_rgb)  # get pillow image
                        draw = ImageDraw.Draw(pil_im)  # create draw object for text
                        font = ImageFont.truetype(
                            os.path.split(__file__)[0] + "/bin/arial.ttf", 60
                        )  # use arial font
                        draw.multiline_text(
                            (170, 40), text, font=font, fill=(0, 0, 0, 255), spacing=10
                        )  # draw the text

                        info = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
                        im_v = cv2.vconcat([info, mah, isi])
                        im_all = cv2.hconcat([wf, im_v])
                        cv2.imwrite(
                            finalpath + "/Cluster_" + str(cluster) + ".png", im_all
                        )  # save the image
            except Exception as e:
                print(
                    "Could not create superplots for channel "
                    + channel
                    + ". Encountered the following error: "
                    + str(e)
                )

    def compile_isoi(self, maxclust=7, Lrat_cutoff=0.1):
        path = self.dir_manager.reports / "clusters"
        file_isoi = pd.DataFrame()
        errorfiles = pd.DataFrame(columns=["channel", "solution", "file"])
        for channel in os.listdir(path):
            channel_isoi = pd.DataFrame()
            for soln in range(3, maxclust + 1):
                try:
                    channel_isoi = channel_isoi.append(
                        pd.read_csv(
                            path + "/{}/clusters{}/isoinfo.csv".format(channel, soln)
                        )
                    )
                except Exception as e:
                    print(e)
                    errorfiles = errorfiles.append(
                        [
                            {
                                "channel": channel[-1],
                                "solution": soln,
                                "file": os.path.split(path)[0],
                            }
                        ]
                    )
            channel_isoi.to_csv(
                "{}/{}/{}_iso_info.csv".format(path, channel, channel), index=False
            )  # output data for the whole channel to the proper folder
            file_isoi = file_isoi.append(
                channel_isoi
            )  # add this channel's info to the whole file info
            try:
                file_isoi = file_isoi.drop(columns=["Unnamed: 0"])
            except:
                pass
        with pd.ExcelWriter(
            os.path.split(path)[0] + f"/{os.path.split(path)[-1]}_compiled_isoi.xlsx",
            engine="xlsxwriter",
        ) as outwrite:
            file_isoi.to_excel(outwrite, sheet_name="iso_data", index=False)
            if (
                errorfiles.size == 0
            ):  # if there are no error csv's add some nans and output to the Excel
                errorfiles = errorfiles.append(
                    [{"channel": "nan", "solution": "nan", "file": "nan"}]
                )
            errorfiles.to_excel(outwrite, sheet_name="errors")
            workbook = outwrite.book
            worksheet = outwrite.sheets["iso_data"]
            redden = workbook.add_format({"bg_color": "red"})
            orangen = workbook.add_format({"bg_color": "orange"})
            yellen = workbook.add_format({"bg_color": "yellow"})
            # add conditional formatting based on ISI's
            worksheet.conditional_format(
                "A2:H{}".format(file_isoi.shape[0] + 1),
                {
                    "type": "formula",
                    "criteria": "=AND($G2>1,$H2>{})".format(str(Lrat_cutoff)),
                    "format": redden,
                },
            )
            worksheet.conditional_format(
                f"A2:H{file_isoi.shape[0] + 1}",
                {
                    "type": "formula",
                    "criteria": f"=OR(AND($G2>.5,$H2>{str(Lrat_cutoff)}),$G2>1)",
                    "format": orangen,
                },
            )
            worksheet.conditional_format(
                "A2:H{}".format(file_isoi.shape[0] + 1),
                {
                    "type": "formula",
                    "criteria": f"=OR($G2>.5,$H2>{str(Lrat_cutoff)})",
                    "format": yellen,
                },
            )
            outwrite.save()
