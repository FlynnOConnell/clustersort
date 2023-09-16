# -*- coding: utf-8 -*-
from __future__ import annotations
import configparser
import os
import shutil
import time
import traceback
import warnings
from datetime import date
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ImageFont, ImageDraw, Image
from matplotlib import cm
from scipy import linalg
from scipy.interpolate import interp1d
from scipy.spatial.distance import mahalanobis

from spk2py import autosort as clust


def infofile(filename, path, sort_time, params):
    # dumps run info to a .info file
    config = configparser.ConfigParser()
    config['METADATA'] = {
        'h5 File': filename, 'Run Time': sort_time,
        'Run Date': date.today().strftime("%m/%d/%y")
      }
    config['PARAMS USED'] = params
    with open(path + '/' + os.path.splitext(filename)[0] + '_' + 'sort.info', 'w') as info_file:
        config.write(info_file)


def process(
        filename: Path | str,
        data: np.ndarray,
        temp_path: Path | str,
        chan_num: int,
        params,
):
    retried = 0
    while True:
        try:
            filename = Path(filename).resolve()

            # Replace any existing paths to flush the old data
            paths_to_check = [
                filename.parent / "Plots" / str(chan_num + 1),
                filename.parent / "spike_waveforms" / f"channel {chan_num + 1}",
                filename.parent / "spike_times" / f"channel {chan_num + 1}",
                filename.parent / "clustering_results" / f"channel {chan_num + 1}",
            ]

            for path in paths_to_check:
                if path.is_dir():
                    shutil.rmtree(path)
                path.mkdir(parents=True, exist_ok=True)

            # Assign the parameters to variables
            max_clusters = int(params['max_clusters'])
            num_iter = int(params['max_iterations'])
            thresh = float(params['convergence_criterion'])
            num_restarts = int(params['random_restarts'])
            wf_amplitude_sd_cutoff = float(params['intra_cluster_cutoff'])
            sampling_rate = float(params['sampling_rate'])
            cutoff_std = float(params['artifact_removal'])
            pvar = float(params['variance_explained'])
            usepvar = int(params['use_percent_variance'])
            userpc = int(params['principal_component_n'])

            spikes = data['wavedata']
            chan_dir = filename.parent / "Plots" / str(chan_num + 1)
            clustering_results_dir = filename.parent / "clustering_results" / f"channel {chan_num + 1}"
            if spikes.size == 0:
                (chan_dir / 'no_spikes.txt').write_text(
                    'No spikes were found on this channel.'
                    ' The most likely cause is an early recording cutoff.'
                )
                warnings.warn(
                    'No spikes were found on this channel. The most likely cause is an early recording cutoff.'
                )
                (clustering_results_dir / 'success.txt').write_text('Sorting finished. No spikes found')
                return
            spikes_final = []
            xnew = np.linspace(0, len(spikes[0]) - 1, len(spikes[0]) * 10)
            slice_cutoff = np.std(spikes) * cutoff_std
            for i in range(len(spikes)):  # this loops through each spike and interpolates the waveform
                if np.any(np.absolute(spikes[i]) > slice_cutoff):
                    continue
                f = interp1d(np.arange(0, len(spikes[0]), 1), spikes[i])
                ynew = f(xnew)
                spikes_final.append(ynew)
            spikes_final = np.array(spikes_final)
            del xnew, f, ynew, spikes

            # Dejitter these spike waveforms, and get their maximum amplitudes
            amplitudes = np.min(spikes_final, axis=1)

            # Save these spikes/spike waveforms and their times to their respective folders
            np.save(os.path.normpath(
                filename[:-3] + '/spike_waveforms/channel {}/spike_waveforms.npy'.format(chan_num + 1)),
                spikes_final)
            np.save(os.path.normpath(
                filename[:-3] + '/spike_times/channel {}/spike_times.npy'.format(chan_num + 1)))

            # Scale the dejittered spikes by the energy of the waveforms
            scaled_slices, energy = clust.scale_waveforms(spikes_final)

            # Run PCA on the scaled waveforms
            pca_slices, explained_variance_ratio = clust.implement_pca(scaled_slices)

            # get cumulative variance explained
            cumulvar = np.cumsum(explained_variance_ratio)
            graphvar = list(cumulvar[0:np.where(cumulvar > .999)[0][0] + 1])

            if usepvar == 1:
                n_pc = np.where(cumulvar > pvar)[0][0] + 1
            else:
                n_pc = userpc

            # Save the pca_slices, energy and amplitudes to the spike_waveforms folder for this channel
            np.save(os.path.normpath(
                filename[:-3] + '/spike_waveforms/channel {}/pca_waveforms.npy'.format(chan_num + 1)),
                pca_slices)
            np.save(
                os.path.normpath(filename[:-3] + '/spike_waveforms/channel {}/energy.npy'.format(chan_num + 1)),
                energy)
            np.save(os.path.normpath(
                filename[:-3] + '/spike_waveforms/channel {}/spike_amplitudes.npy'.format(chan_num + 1)),
                amplitudes)

            # Create file for saving plots, and plot explained variance ratios of the PCA
            fig = plt.figure()
            x = np.arange(0, len(graphvar) + 1)
            graphvar.insert(0, 0)
            plt.plot(x, graphvar)
            plt.vlines(n_pc, 0, 1, colors='r')
            plt.annotate(
                str(n_pc) + " PC's used for GMM.\nVariance explained= " + str(round(cumulvar[n_pc - 1], 3)) + "%.",
                (n_pc + .25, cumulvar[n_pc - 1] - .1))
            plt.title('Variance ratios explained by PCs (cumulative)')
            plt.xlabel('PC #')
            plt.ylabel('Explained variance ratio')
            fig.savefig(os.path.normpath(filename[:-3] + '/Plots/{}/pca_variance.png'.format(chan_num + 1)),
                        bbox_inches='tight')
            plt.close("all")

            # Make an array of the data to be used for clustering, and delete pca_slices, scaled_slices, energy and
            # amplitudes
            data = np.zeros((len(pca_slices), n_pc + 2))
            data[:, 2:] = pca_slices[:, :n_pc]
            data[:, 0] = energy[:] / np.max(energy)
            data[:, 1] = np.abs(amplitudes) / np.max(np.abs(amplitudes))
            del pca_slices
            del scaled_slices
            del energy
            break
        except MemoryError:
            if retried == 1:
                traceback.print_exc()
                return
            warnings.warn(
                f"Warning, could not allocate memory for channel {chan_num + 1}. This program will wait and try "
                f"again in a bit."
            )
            retried = 1
            time.sleep(1200)
        except:
            traceback.print_exc()
            return

    # Run GMM, from 3 to max_clusters
    for i in range(max_clusters - 2):
        # print("Creating PCA plots.")
        try:
            model, predictions, bic = clust.clusterGMM(data, n_clusters=i + 3, n_iter=num_iter, restarts=num_restarts,
                                                       threshold=thresh)
        except:
            # print "Clustering didn't work - solution with %i clusters most likely didn't converge" % (i+3)
            continue
        if np.any([len(np.where(predictions[:] == cluster)[0]) <= n_pc + 2 for cluster in range(i + 3)]):
            os.mkdir(filename[:-3] + '/Plots/%i/%i_clusters' % ((chan_num + 1), i + 3))
            os.mkdir(filename[:-3] + '/Plots/%i/%i_clusters_waveforms_ISIs' % ((chan_num + 1), i + 3))
            with open(filename[:-3] + '/Plots/%i/%i_clusters' % ((chan_num + 1), i + 3) + '/invalid_sort.txt',
                      "w+") as f:
                f.write("There are too few waveforms to properly sort this clustering")
            with open(filename[:-3] + '/Plots/%i/%i_clusters_waveforms_ISIs' % (
                    (chan_num + 1), i + 3) + '/invalid_sort.txt', "w+") as f:
                f.write("There are too few waveforms to properly sort this clustering")
            continue
        # Sometimes large amplitude noise waveforms hpc_cluster with the spike waveforms because the amplitude has
        # been factored out of the scaled spikes. Run through the clusters and find the waveforms that are more than
        # wf_amplitude_sd_cutoff larger than the hpc_cluster mean. Set predictions = -1 at these points so that they
        # aren't picked up by Pl2_PostProcess
        for cluster in range(i + 3):
            cluster_points = np.where(predictions[:] == cluster)[0]
            this_cluster = predictions[cluster_points]
            cluster_amplitudes = amplitudes[cluster_points]
            cluster_amplitude_mean = np.mean(cluster_amplitudes)
            cluster_amplitude_sd = np.std(cluster_amplitudes)
            reject_wf = \
                np.where(cluster_amplitudes <= cluster_amplitude_mean - wf_amplitude_sd_cutoff * cluster_amplitude_sd)[
                    0]
            this_cluster[reject_wf] = -1
            predictions[cluster_points] = this_cluster

            # Make folder for results of i+2 clusters, and store results there
        os.mkdir(filename[:-3] + '/clustering_results/channel %i/clusters%i' % ((chan_num + 1), i + 3))
        np.save(os.path.normpath(
            filename[:-3] + '/clustering_results/channel {}/clusters{}/predictions.npy'.format(chan_num + 1,
                                                                                                  i + 3)), predictions)
        np.save(os.path.normpath(
            filename[:-3] + '/clustering_results/channel {}/clusters{}/bic.npy'.format(chan_num + 1, i + 3)),
            bic
        )

        # Plot the graphs, for this set of clusters, in the directory made for this channel
        os.mkdir(filename[:-3] + '/Plots/%i/%i_clusters' % ((chan_num + 1), i + 3))
        colors = cm.rainbow(np.linspace(0, 1, i + 3))

        for feature1 in range(len(data[0])):
            for feature2 in range(len(data[0])):
                if feature1 < feature2:
                    fig = plt.figure()
                    plt_names = []
                    for cluster in range(i + 3):
                        plot_data = np.where(predictions[:] == cluster)[0]
                        plt_names.append(
                            plt.scatter(data[plot_data, feature1], data[plot_data, feature2], color=colors[cluster],
                                        s=0.8))

                    plt.xlabel("Feature %i" % feature1)
                    plt.ylabel("Feature %i" % feature2)
                    # Produce figure legend
                    plt.legend(tuple(plt_names), tuple("Cluster %i" % cluster for cluster in range(i + 3)),
                               scatterpoints=1, loc='lower left', ncol=3, fontsize=8)
                    plt.title("%i clusters" % (i + 3))
                    fig.savefig(os.path.normpath(
                        filename[:-3] + '/Plots/{}/{}_clusters/feature{}vs{}.png'.format(chan_num + 1, i + 3,
                                                                                          feature2, feature1)))
                    plt.close("all")

        for ref_cluster in range(i + 3):
            fig = plt.figure()
            ref_mean = np.mean(data[np.where(predictions == ref_cluster)], axis=0)
            ref_covar_I = linalg.inv(np.cov(data[np.where(predictions == ref_cluster)[0]], rowvar=False))
            xsave = None
            for other_cluster in range(i + 3):
                mahalanobis_dist = [mahalanobis(data[point, :], ref_mean, ref_covar_I) for point in
                                    np.where(predictions[:] == other_cluster)[0]]
                # Plot histogram of Mahalanobis distances
                y, binEdges = np.histogram(mahalanobis_dist, bins=25)
                bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
                plt.plot(bincenters, y, label='Dist from hpc_cluster %i' % other_cluster)
                if other_cluster == ref_cluster:
                    xsave = bincenters

            plt.xlim([0, max(xsave) + 5])
            plt.xlabel('Mahalanobis distance')
            plt.ylabel('Frequency')
            plt.legend(loc='upper right', fontsize=8)
            plt.title('Mahalanobis distance of all clusters from Reference Cluster: %i' % ref_cluster)
            fig.savefig(os.path.normpath(
                filename[:-3] + '/Plots/{}/{}_clusters/Mahalonobis_cluster{}.png'.format(chan_num + 1, i + 3,
                                                                                          ref_cluster)))
            plt.close("all")

        # Create file, and plot spike waveforms for the different clusters. Plot 10 times downsampled
        # dejittered/smoothed waveforms. Plot the ISI distribution of each hpc_cluster
        os.mkdir(filename[:-3] + '/Plots/%i/%i_clusters_waveforms_ISIs' % ((chan_num + 1), i + 3))
        ISIList = []
        for cluster in range(i + 3):
            cluster_points = np.where(predictions[:] == cluster)[0]
            fig, ax = plt.subplots()
            # fig, ax = AutoSort.Pl2_waveforms_datashader.waveforms_datashader(
            #     spikes_final[cluster_points, :],
            #     x,
            #     dir_name=os.path.normpath(filename[:-3] + "_datashader_temp_el" + str(chan_num + 1))
            # )
            ax.set_xlabel('Sample ({:d} samples per ms)'.format(int(sampling_rate / 1000)))
            ax.set_ylabel('Voltage (microvolts)')
            ax.set_title('Cluster%i' % cluster)
            fig.savefig(os.path.normpath(
                filename[:-3] + '/Plots/{}/{}_clusters_waveforms_ISIs/Cluster{}_waveforms'.format(chan_num + 1,
                                                                                                   i + 3, cluster)))
            plt.close("all")

            fig = plt.figure()
            cluster_times = times_final[cluster_points]
            ISIs = np.ediff1d(np.sort(cluster_times))
            ISIs = ISIs / 40.0
            plt.hist(ISIs, bins=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, np.max(ISIs)])
            plt.xlim([0.0, 10.0])
            plt.title("2ms ISI violations = %.1f percent (%i/%i)" % (
                (float(len(np.where(ISIs < 2.0)[0])) / float(len(cluster_times))) * 100.0, len(np.where(ISIs < 2.0)[0]),
                len(cluster_times)) + '\n' + "1ms ISI violations = %.1f percent (%i/%i)" % (
                          (float(len(np.where(ISIs < 1.0)[0])) / float(len(cluster_times))) * 100.0,
                          len(np.where(ISIs < 1.0)[0]), len(cluster_times)))
            fig.savefig(os.path.normpath(
                filename[:-3] + '/Plots/{}/{}_clusters_waveforms_ISIs/Cluster{}_ISIs'.format(chan_num + 1, i + 3,
                                                                                              cluster)))
            plt.close("all")
            ISIList.append("%.1f" % ((float(len(np.where(ISIs < 1.0)[0])) / float(len(cluster_times))) * 100.0))

            # Get isolation statistics for each solution
        Lrats = clust.get_Lratios(data, predictions)
        isodf = pd.DataFrame({
            'IsoRating': 'TBD',
            'File': os.path.split(filename[:-3])[-1],
            'Channel': chan_num + 1,
            'Solution': i + 3,
            'Cluster': range(i + 3),
            'wf count': [len(np.where(predictions[:] == cluster)[0]) for cluster in range(i + 3)],
            'ISIs (%)': ISIList,
            'L-Ratio': [round(Lrats[cl], 3) for cl in range(i + 3)],
        })
        isodf.to_csv(os.path.splitext(filename)[0] + '/clustering_results/channel {}/clusters{}/isoinfo.csv'.format(
            chan_num + 1, i + 3), index=False)
        # output this all in a plot in the plots folder and replace the ISI plot in superplots
        for cluster in range(i + 3):
            text = 'wf count: \n1 ms ISIs: \nL-Ratio: '  # package text to be plotted
            text2 = '{}\n{}%\n{}'.format(isodf['wf count'][cluster], isodf['ISIs (%)'][cluster],
                                         isodf['L-Ratio'][cluster])
            blank = np.ones((480, 640, 3), np.uint8) * 255  # initialize empty whihte image
            cv2_im_rgb = cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)  # convert to color space pillow can use
            pil_im = Image.fromarray(cv2_im_rgb)  # get pillow image
            draw = ImageDraw.Draw(pil_im)  # create draw object for text
            font = ImageFont.truetype(os.path.split(__file__)[0] + "/bin/arial.ttf", 60)  # use arial font
            draw.multiline_text((90, 100), text, font=font, fill=(0, 0, 0, 255), spacing=50,
                                align='left')  # draw the text
            draw.multiline_text((380, 100), text2, font=font, fill=(0, 0, 0, 255), spacing=50)  # draw the text
            isoimg = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)  # convert back to openCV image
            cv2.imwrite(filename[:-3] + '/Plots/{}/{}_clusters_waveforms_ISIs/Cluster{}_Isolation.png'.format(
                chan_num + 1, i + 3, cluster), isoimg)  # save the image
    with open(filename[:-3] + '/clustering_results/channel {}'.format(chan_num + 1) + '/success.txt',
              'w+') as f:
        f.write('Congratulations, this channel was sorted successfully')


def superplots(full_filename, maxclust):
    # This function takes all the plots and conglomerates them
    path = os.path.splitext(full_filename)[0] + '/Plots'  # The path holding the plots to be run on
    outpath = os.path.splitext(full_filename)[0] + '/superplots'  # output path for superplots
    if os.path.isdir(outpath):  # if the path for plots exists remove it
        shutil.rmtree(outpath)
    os.mkdir(outpath)  # make the output path
    for channel in os.listdir(path):  # for each channel
        try:
            currentpath = path + '/' + channel
            os.mkdir(outpath + '/' + channel)  # create an output path for each channel
            for soln in range(3, maxclust + 1):  # for each number hpc_cluster solution
                finalpath = outpath + '/' + channel + '/' + str(soln) + '_clusters'
                os.mkdir(finalpath)  # create output folders
                for cluster in range(0, soln):  # for each hpc_cluster
                    mah = cv2.imread(
                        currentpath + '/' + str(soln) + '_clusters/Mahalonobis_cluster' + str(cluster) + '.png')
                    if not np.shape(mah)[0:2] == (480, 640):
                        mah = cv2.resize(mah, (640, 480))
                    wf = cv2.imread(currentpath + '/' + str(soln) + '_clusters_waveforms_ISIs/Cluster' + str(
                        cluster) + '_waveforms.png')
                    if not np.shape(mah)[0:2] == (1200, 1600):
                        wf = cv2.resize(wf, (1600, 1200))
                    isi = cv2.imread(currentpath + '/' + str(soln) + '_clusters_waveforms_ISIs/Cluster' + str(
                        cluster) + '_Isolation.png')
                    if not np.shape(isi)[0:2] == (480, 640):
                        isi = cv2.resize(isi, (640, 480))
                    blank = np.ones((240, 640, 3), np.uint8) * 255  # make whitespace for info
                    text = "Electrode: " + channel + "\nSolution: " + str(soln) + "\nCluster: " + str(
                        cluster)  # text to output to whitespace (hpc_cluster, channel, and solution numbers)
                    cv2_im_rgb = cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)  # convert to color space pillow can use
                    pil_im = Image.fromarray(cv2_im_rgb)  # get pillow image
                    draw = ImageDraw.Draw(pil_im)  # create draw object for text
                    font = ImageFont.truetype(os.path.split(__file__)[0] + "/bin/arial.ttf", 60)  # use arial font
                    draw.multiline_text((170, 40), text, font=font, fill=(0, 0, 0, 255), spacing=10)  # draw the text
                    info = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)  # convert back to openCV image
                    im_v = cv2.vconcat([info, mah, isi])  # concatenate images together
                    im_all = cv2.hconcat([wf, im_v])  # continued concatenation
                    cv2.imwrite(finalpath + '/Cluster_' + str(cluster) + '.png', im_all)  # save the image
        except Exception as e:
            print("Could not create superplots for channel " + channel + ". Encountered the following error: " + str(e))


def compile_isoi(full_filename, maxclust=7, Lrat_cutoff=.1):
    path = os.path.splitext(full_filename)[0] + '/clustering_results'
    file_isoi = pd.DataFrame()
    errorfiles = pd.DataFrame(columns=['channel', 'solution', 'file'])
    for channel in os.listdir(path):  # for each channel
        channel_isoi = pd.DataFrame()
        for soln in range(3, maxclust + 1):  # for each solution
            try:  # get the isoi info for this solution and add it to the channel data
                channel_isoi = channel_isoi.append(
                    pd.read_csv(path + '/{}/clusters{}/isoinfo.csv'.format(channel, soln)))
            except Exception as e:  # if an error occurs, add it to the list of error files
                print(e)
                errorfiles = errorfiles.append(
                    [{'channel': channel[-1], 'solution': soln, 'file': os.path.split(path)[0]}])
        channel_isoi.to_csv('{}/{}/{}_iso_info.csv'.format(path, channel, channel),
                            index=False)  # output data for the whole channel to the proper folder
        file_isoi = file_isoi.append(channel_isoi)  # add this channel's info to the whole file info
        try:
            file_isoi = file_isoi.drop(columns=['Unnamed: 0'])
        except:
            pass
    with pd.ExcelWriter(os.path.split(path)[0] + f'/{os.path.split(path)[-1]}_compiled_isoi.xlsx',
                        engine='xlsxwriter') as outwrite:
        file_isoi.to_excel(outwrite, sheet_name='iso_data', index=False)
        if errorfiles.size == 0:  # if there are no error csv's add some nans and output to the Excel
            errorfiles = errorfiles.append([{'channel': 'nan', 'solution': 'nan', 'file': 'nan'}])
        errorfiles.to_excel(outwrite, sheet_name='errors')
        workbook = outwrite.book
        worksheet = outwrite.sheets['iso_data']
        redden = workbook.add_format({'bg_color': 'red'})
        orangen = workbook.add_format({'bg_color': 'orange'})
        yellen = workbook.add_format({'bg_color': 'yellow'})
        # add conditional formatting based on ISI's
        worksheet.conditional_format('A2:H{}'.format(file_isoi.shape[0] + 1),
                                     {'type': 'formula', 'criteria': '=AND($G2>1,$H2>{})'.format(str(Lrat_cutoff)),
                                      'format': redden})
        worksheet.conditional_format(
            f'A2:H{file_isoi.shape[0] + 1}',
            {
                'type': 'formula',
                'criteria': f'=OR(AND($G2>.5,$H2>{str(Lrat_cutoff)}),$G2>1)',
                'format': orangen
            })
        worksheet.conditional_format('A2:H{}'.format(file_isoi.shape[0] + 1),
                                     {
                                         'type': 'formula',
                                         'criteria': f'=OR($G2>.5,$H2>{str(Lrat_cutoff)})',
                                         'format': yellen
                                     }
                                     )
        outwrite.save()
