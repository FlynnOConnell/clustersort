# -*- coding: utf-8 -*-
from __future__ import annotations

import configparser
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
from spk2py import autosort as clust
from .utils import excepts

logger = logging.getLogger(__name__)
logpath = Path().home() / "autosort" / "directory_logs.log"
logging.basicConfig(filename=logpath, level=logging.DEBUG)
logger.addHandler(logging.StreamHandler())


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
        self.params = params
        self.chan_num = chan_num
        self.dir_manager = dir_manager

    @property
    def pvar(self):
        return self.params['variance_explained']

    @property
    def usepvar(self):
        return int(self.params['use_percent_variance'])

    @property
    def userpc(self):
        return int(self.params['principal_component_n'])

    @property
    def max_clusters(self):
        return int(self.params['max_clusters'])

    @property
    def max_iterations(self):
        return int(self.params['max_iterations'])

    @property
    def thresh(self):
        return float(self.params['convergence_criterion'])

    @property
    def num_restarts(self):
        return int(self.params['random_restarts'])

    @property
    def wf_amplitude_sd_cutoff(self):
        return float(self.params['intra_cluster_cutoff'])

    @property
    def sampling_rate(self):
        return float(self.params['sampling_rate'])

    @property
    def artifact_removal(self):
        return float(self.params['artifact_removal'])

    @property
    def pre_time(self):
        return float(self.params['pre_time'])

    @property
    def post_time(self):
        return float(self.params['post_time'])

    @property
    def bandpass(self):
        return float(self.params['low_cutoff']), float(self.params['high_cutoff'])

    @property
    def spike_detection(self):
        return int(self.params['spike_detection'])

    @property
    def STD(self):
        return int(self.params['spike_detection'])

    @property
    def cutoff_std(self):
        return float(self.params['artifact_removal'])

    @property
    def max_breach_rate(self):
        return float(self.params['max_breach_rate'])

    @property
    def max_breach_count(self):
        return float(self.params['max_breach_count'])

    @property
    def max_breach_avg(self):
        return float(self.params['max_breach_avg'])

    @property
    def intra_cluster_cutoff(self):
        return float(self.params['intra_cluster_cutoff'])

    @property
    def voltage_cutoff(self):
        return float(self.params['voltage_cutoff'])


    def process_continuous(self):

        filt_el = clust.filter_signal(
            self.data['wavedata'],
            freq=(self.bandpass[0], self.bandpass[1]),
            sampling_rate=self.sampling_rate,
        )
        breach_rate = float(
            len(np.where(filt_el > self.voltage_cutoff)[0]) * int(self.sampling_rate)
        ) / len(filt_el)

        test_el = np.reshape(
            filt_el[: int(self.sampling_rate) * int(len(filt_el) / self.sampling_rate)],
            (-1, int(self.sampling_rate)),
        )

        breaches_per_sec = [
            len(np.where(test_el[i] > self.voltage_cutoff)[0]) for i in range(len(test_el))
        ]
        breaches_per_sec = np.array(breaches_per_sec)
        secs_above_cutoff = len(np.where(breaches_per_sec > 0)[0])
        if secs_above_cutoff == 0:
            mean_breach_rate_persec = 0
        else:
            mean_breach_rate_persec = np.mean(
                breaches_per_sec[np.where(breaches_per_sec > 0)[0]]
            )

        if (
                breach_rate >= self.max_breach_rate
                and secs_above_cutoff >= self.max_breach_count
                and mean_breach_rate_persec >= self.max_breach_avg
        ):
            recording_cutoff = np.where(breaches_per_sec > self.max_breach_avg)[0][0]

            # Then cut the recording accordingly
            filt_el = filt_el[: recording_cutoff * int(self.sampling_rate)]

        if len(filt_el) == 0:
            slices, spike_times = [], []
        else:
            slices, spike_times = clust.extract_waveforms(
                filt_el,
                spike_snapshot=[self.pre_time, self.post_time],
                sampling_rate=self.sampling_rate,
                STD=self.STD,
                cutoff_std=self.cutoff_std,
            )

        if len(slices) == 0 or len(spike_times) == 0:
            with open(
                    self.dir_manager.temp_path / "/Plots/" + str(self.chan_num + 1) + "/" + "no_spikes.txt", "w"
            ) as txt:
                txt.write(
                    "No spikes were found on channel {}. The most likely cause is an early recording cutoff. RIP".format(
                        self.chan_num + 1
                    )
                )
                warnings.warn(
                    "No spikes were found on channel {}. The most likely cause is an early recording cutoff. RIP".format(
                        self.chan_num + 1
                    )
                )
                with open(
                    self.dir_manager / "clustering_results" / f"channel_{self.chan_num + 1}" / "success.txt", "w+"
                ) as f:
                    f.write("Sorting finished. No spikes found")
                return None

        slices, times = clust.dejitter(
            slices,
            spike_times,
            spike_snapshot=(self.pre_time, self.post_time),
            sampling_rate=self.sampling_rate,
        )

        return slices, times


    def process_channel(
        self,
        dir_manager: DirectoryManager,
        chan_num: int,
    ):
        while True:
            if self.data['wavedata'].size == 0:
                (
                    dir_manager.reports / f"channel_{chan_num + 1}" / "no_spikes.txt"
                ).write_text(
                    "No spikes were found on this channel."
                    " The most likely cause is an early recording cutoff."
                )
                warnings.warn(
                    "No spikes were found on this channel. The most likely cause is an early recording cutoff."
                )
                (
                    dir_manager.reports / f"channel_{chan_num + 1}" / "success.txt"
                ).write_text("Sorting finished. No spikes found")
                return
            else:
                spikes_final, times_final = self.process_continuous()

            # Dejitter these spike waveforms, and get their maximum amplitudes
            amplitudes = np.min(spikes_final, axis=1)

            np.save(
                dir_manager.intermediate
                / f"channel_{chan_num + 1}"
                / "spike_waveforms.npy",
                spikes_final,
            )
            np.save(
                dir_manager.intermediate / f"channel_{chan_num + 1}" / "spike_times.npy",
                times_final,
            )

            # Scale the dejittered spikes by the energy of the waveforms and perform PCA
            scaled_slices, energy = clust.scale_waveforms(spikes_final)
            pca_slices, explained_variance_ratio = clust.implement_pca(scaled_slices)
            cumulvar = np.cumsum(explained_variance_ratio)
            graphvar = list(cumulvar[0 : np.where(cumulvar > 0.999)[0][0] + 1])

            if self.usepvar == 1:
                n_pc = np.where(cumulvar > self.pvar)[0][0] + 1
            else:
                n_pc = self.userpc

            np.save(
                dir_manager.intermediate
                / f"channel_{chan_num + 1}"
                / "spike_waveforms.npy",
                spikes_final,
            )
            np.save(
                dir_manager.intermediate / f"channel_{chan_num + 1}" / "spike_times.npy",
                times_final,
            )
            np.save(
                dir_manager.intermediate
                / f"channel_{chan_num + 1}"
                / "spike_waveforms_pca.npy",
                pca_slices,
            )

            # explained variance
            var = float(cumulvar[n_pc - 1])  # mainly to avoid type checking issues in the annotation below
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
                dir_manager.plots / f"channel_{chan_num + 1}" / "pca_variance.png",
                bbox_inches="tight",
            )
            plt.close("all")

            # Make an array of the data to be used for clustering
            data = np.zeros((len(pca_slices), n_pc + 2))
            data[:, 2:] = pca_slices[:, :n_pc]
            data[:, 0] = energy[:] / np.max(energy)
            data[:, 1] = np.abs(amplitudes) / np.max(np.abs(amplitudes))
            break


    def spk_gmm(self, data, times_final, chan_num, n_pc, amplitudes):
        for i in range(self.max_clusters - 2):
            try:
                model, predictions, bic = clust.clusterGMM(
                    data,
                    n_clusters=i + 3,
                    n_iter=self.max_iterations,
                    restarts=self.num_restarts,
                    threshold=self.thresh,
                )
            except Exception as e:
                logger.debug("Error in clusterGMM", exc_info=True)
                continue

            if np.any(
                [
                    len(np.where(predictions[:] == cluster)[0]) <= n_pc + 2
                    for cluster in range(i + 3)
                ]
            ):
                plots_waveforms_ISIs_path = (
                    self.dir_manager.plots
                    / f"channel_{chan_num + 1}/{i + 3}_clusters_waveforms_ISIs"
                )
                plots_waveforms_ISIs_path.mkdir(parents=True, exist_ok=True)
        
                # Create and write to the invalid_sort.txt files
                with open(
                    self.dir_manager.plots / f"channel_{chan_num + 1}" / "invalid_sort.txt", "w+"
                ) as f:
                    f.write("There are too few waveforms to properly sort this clustering")
        
                with open(
                    self.dir_manager.plots / f"channel_{chan_num + 1}" / "invalid_sort.txt", "w+"
                ) as f:
                    f.write("There are too few waveforms to properly sort this clustering")
        
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
                    <= cluster_amplitude_mean - self.wf_amplitude_sd_cutoff * cluster_amplitude_sd
                )[0]
                this_cluster[reject_wf] = -1
                predictions[cluster_points] = this_cluster
        
                # Make folder for results of i+2 clusters, and store results there
                clusters_path = (
                    self.dir_manager.plots
                    / f"clustering_results/channel_{chan_num + 1}/clusters{i + 3}"
                )
                clusters_path.mkdir(parents=True, exist_ok=True)
        
                np.save(clusters_path / "predictions.npy", predictions)
                np.save(clusters_path / "bic.npy", bic)
        
                # Plot the graphs, for this set of clusters, in the directory made for this channel
                plots_path = self.dir_manager.plots / f"channel_{chan_num + 1}/{i + 3}_clusters"
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
                            / f"channel_{chan_num + 1}/{i + 3}_clusters/feature{feature2}vs{feature1}.png",
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
                    plt.plot(bincenters, y, label="Dist from hpc_cluster %i" % other_cluster)
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
                    / f"channel_{chan_num + 1}/{i + 3}_clusters/Mahalonobis_cluster{ref_cluster}.png",
                )
                plt.close("all")
        
            # Create file, and plot spike waveforms for the different clusters. Plot 10 times downsampled
            # dejittered/smoothed waveforms. Plot the ISI distribution of each hpc_cluster
            for cluster in range(i + 3):
                clust_path = (
                    self.dir_manager.plots
                    / f"channel_{chan_num + 1}/{i + 3}_clusters_waveforms_ISIs"
                )
                clust_path.mkdir(parents=True, exist_ok=True)
        
            ISIList = []
            for cluster in range(i + 3):
                cluster_points = np.where(predictions[:] == cluster)[0]
                fig, ax = plt.subplots()
                # fig, ax = waveforms_datashader(
                #     spikes_final[cluster_points, :],
                #     x,
                #     self.dir_manager.filename + "_datashader_temp_el" + str(chan_num + 1),
                # )
                ax.set_xlabel("Sample ({:d} samples per ms)".format(int(self.sampling_rate / 1000)))
                ax.set_ylabel("Voltage (microvolts)")
                ax.set_title("Cluster%i" % cluster)
                fig.savefig(
                    self.dir_manager.plots
                    / f"channel_{chan_num + 1}/{i + 3}_clusters_waveforms_ISIs/Cluster{cluster}_waveforms"
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
                        (float(len(np.where(ISIs < 2.0)[0])) / float(len(cluster_times)))
                        * 100.0,
                        len(np.where(ISIs < 2.0)[0]),
                        len(cluster_times),
                    )
                    + "\n"
                    + "1ms ISI violations = %.1f percent (%i/%i)"
                    % (
                        (float(len(np.where(ISIs < 1.0)[0])) / float(len(cluster_times)))
                        * 100.0,
                        len(np.where(ISIs < 1.0)[0]),
                        len(cluster_times),
                    )
                )
                fig.savefig(
                    self.dir_manager.plots
                    / f"channel_{chan_num + 1}/{i + 3}_clusters_waveforms_ISIs/Cluster{cluster}_ISIs"
                )
                plt.close("all")
                ISIList.append(
                    "%.1f"
                    % (
                        (float(len(np.where(ISIs < 1.0)[0])) / float(len(cluster_times)))
                        * 100.0
                    )
                )
        
            # Get isolation statistics for each solution
            Lrats = clust.get_Lratios(data, predictions)
        
            isodf = pd.DataFrame(
                {
                    "IsoRating": "TBD",
                    "File": self.dir_manager.filename,
                    "Channel": chan_num + 1,
                    "Solution": i + 3,
                    "Cluster": range(i + 3),
                    "wf count": [
                        len(np.where(predictions[:] == cluster)[0]) for cluster in range(i + 3)
                    ],
                    "ISIs (%)": ISIList,
                    "L-Ratio": [round(Lrats[cl], 3) for cl in range(i + 3)],
                }
            )
            cluster_path = self.dir_manager.reports / f"channel_{chan_num + 1}" / f"clusters_{i + 3}"
            cluster_path.mkdir(parents=True, exist_ok=True)
            isodf.to_csv(
                cluster_path / "isoinfo.csv",
                index=False,
            )
            # output this all in a plot in the plots folder and replace the ISI plot in superplots
            for cluster in range(i + 3):
                text = "wf count: \n1 ms ISIs: \nL-Ratio: "  # package text to be plotted
                text2 = "{}\n{}%\n{}".format(
                    isodf["wf count"][cluster],
                    isodf["ISIs (%)"][cluster],
                    isodf["L-Ratio"][cluster],
                )
                blank = np.ones((480, 640, 3), np.uint8) * 255  # initialize empty white image

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
                isoimg = cv2.cvtColor(
                    np.array(pil_im), cv2.COLOR_RGB2BGR
                )
                cv2.imwrite(
                    self.dir_manager.plots
                    / f"channel_{chan_num + 1}"
                    / f"clusters_{i + 3}"
                    / f"cluser_{cluster}_isoimg.png",
                    isoimg,
                )  # save the image
        with open(self.dir_manager.reports / f"channel_{chan_num + 1}" / "success.txt", "w+") as f:
            f.write("Congratulations, this channel was sorted successfully")


    def superplots(self, full_filename, maxclust):
        path = (
            self.dir_manager.plots / f"channel_{self.chan_num + 1}"
        )
        outpath = (
            self.dir_manager.plots / f"channel_{self.chan_num + 1}" / "superplots"
        )
        if outpath.exists():
            shutil.rmtree(outpath)
        outpath.mkdir(parents=True, exist_ok=True)
        for channel in outpath.glob("*"):
            try:
                currentpath = path + "/" + channel
                os.mkdir(outpath + "/" + channel)  # create an output path for each channel
                for soln in range(3, maxclust + 1):  # for each number hpc_cluster solution
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

                        info = cv2.cvtColor(
                            np.array(pil_im), cv2.COLOR_RGB2BGR
                        )
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
                except (
                        Exception
                ) as e:
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