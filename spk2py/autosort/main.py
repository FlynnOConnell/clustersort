# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime
import logging
import math
import multiprocessing
import shutil
from pathlib import Path

import config
from spk2py import io
from autosort import process

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def move_files(files, source, destination):
    for f in files:
        shutil.move(source / f, destination)
        logger.info(f"Moved {f} to {destination}")


def main(default_config=False):
    if default_config:
        params = config.set_config(default=True)
    else:
        params = config.set_config()

    # If the script is being run automatically, on Fridays it will run a greater number of files
    if params['run_type'] == 'Auto':
        if datetime.datetime.weekday(datetime.date.today()) == 4:
            n_files = int(params['weekend_run'])
        else:
            n_files = int(params['weekday_run'])
    elif params['run_type'] == 'Manual':
        n_files = params['manual_run']
    else:
        raise Exception('Run type choice is not valid. Options are "Manual" or "Auto"')

    # Get the files to run
    runpath = Path(params['run_path'])
    runfiles = [f for f in runpath.glob('*.hdf5')][:n_files]

    completed_path = Path(params['completed_path'])
    outpath = Path(params['results_path'])
    num_cpu = int(params['cores_used'])
    resort_limit = int(params['resort_limit'])

    for curr_file in runfiles:  # loop through each file
        file_name = curr_file.name
        file_stem = curr_file.stem
        file_parent = curr_file.parent
        temp_path = file_parent / file_stem / 'temp'  # !TODO: clean later
        temp_path.mkdir(parents=True, exist_ok=True)
        res_dir = completed_path
        res_file = completed_path / file_stem

        h5file = io.h5.read_h5(curr_file)
        num_chan = len(h5file.keys())
        hdf5_dirs = [
            res_file / 'spike_waveforms',
            res_file / 'spike_times',
            res_file / 'clustering_results',
            res_file / 'Plots'
        ]

        for dir_path in hdf5_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        runs = math.ceil(num_chan / num_cpu)
        for n in range(runs):  # For the number of runs
            chan_start = num_cpu * n  # First channel to start with
            chan_end = num_cpu * (n + 1)  # Channel to stop with
            if chan_end > num_chan:
                chan_end = num_chan
            processes = []
            for i in range(chan_start, chan_end):
                # Get the current channel
                this_chan = h5file[list(h5file.keys())[i]]
                p = multiprocessing.Process(target=process, args=(
                    curr_file, this_chan, temp_path, i, params))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
    #     # Integrity check
    #     print("Performing integrity check for sort...")
    #     bad_runs = [9001]
    #     reruns = 0
    #     # while there are still bad runs, and the script has not reached the resort limit
    #     while len(
    #             bad_runs) > 0 and reruns < resort_limit:
    #         reruns += 1
    #         bad_runs = []
    #         for chan in range(1, elNum + 1):  # this plot is the last file the sorting makes, check if it was created
    #             if not os.path.isfile(os.path.splitext(filename)[0] + '/clustering_results/electrode {}'.format(
    #                     str(chan)) + '/success.txt'):
    #                 bad_runs.append(chan)
    #         if len(bad_runs) == 0:
    #             print("All channels were sorted successfully!")
    #         elif len(bad_runs) == 1:  # if there was a bad run, initialize resort
    #             print('Channel', bad_runs[0], 'was not sorted successfully, resorting...')
    #             try:
    #                 AS.Processing(bad_runs[0] - 1, filename, params)
    #             except:
    #                 traceback.print_exc()
    #         else:  # if there were multiple bad runs, parallel resort up to the core count
    #             if len(bad_runs) >= elNum:
    #                 raise Exception("Sorting failed on every channel. Do not close python, talk to Daniel")
    #             if len(bad_runs) > num_cpu:
    #                 bad_runs = bad_runs[0:num_cpu]
    #             print('The following channels:', bad_runs,
    #                   'were not sorted successfully. Initializing parallel processing for resort...')
    #             reprocesses = []
    #             for channel in bad_runs:  # run so many times
    #                 re = multiprocessing.Process(target=AS.Processing, args=(channel - 1, filename,
    #                                                                          params))  # Run Processing function using multiple processes and input argument i
    #                 re.start()  # Start the processes
    #                 reprocesses.append(re)  # catalogue the processes
    #             for re in reprocesses:
    #                 re.join()  # rejoin the individual processes once they are all finished
    #     # superplots
    #     bad_runs = []
    #     for chan in range(1, elNum + 1):  # check again for bad runs
    #         if not os.path.isfile(os.path.splitext(filename)[0] + '/clustering_results/electrode {}'.format(
    #                 str(chan)) + '/success.txt'):
    #             bad_runs.append(chan)
    #     if len(bad_runs) > 0:  # If there are bad runs don't make superplots or isolation compilation
    #         warnings.warn("Warning: Sort unsuccessful on at least one channel!")
    #         print("Sorting failed on the following channels: {}; superplots will not be created.".format(bad_runs))
    #     else:  # else make the superplots
    #         try:
    #             AS.superplots(filename, int(params['max clusters']))
    #         except Exception as e:
    #             warnings.warn("Warning: superplots unsuccessful!")
    #             print(e)
    #         try:
    #             AS.compile_isoi(filename, int(params['max clusters']))  # compile iso data
    #         except Exception as e:
    #             warnings.warn("Warning: isolation information compilation unsuccessful!")
    #             print(e)
    #     sort_time = str(((time.time() - filestart) / 3600))
    #     print("Sort completed.", filename, "ran for", sort_time, "hours.")
    #     AS.infofile(file, os.path.splitext(filename)[0], sort_time, __file__, params)  # create an infofile
    # if usepaths == 1:  # move everything to the completion directories
    #     for file in ranfiles:
    #         shutil.move(running_path + '\\' + file, completed_path + '\\' + file)
    #         shutil.move(running_path + '\\' + os.path.splitext(file)[0] + '.h5',
    #                     outpath + '\\' + os.path.splitext(file)[0] + '.h5')
    #         shutil.move(running_path + '\\' + os.path.splitext(file)[0], outpath + '\\' + os.path.splitext(file)[0])
    # print("Sorting Complete!")


if __name__ == '__main__':
    main(default_config=True)
