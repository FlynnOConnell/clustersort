# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime
import logging
import math
import multiprocessing
import shutil
from pathlib import Path

import spk_config
from spk2py import spk_io
from autosort import run_spk_process
from directory_manager import DirectoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def move_files(files, source, destination):
    for f in files:
        shutil.move(source / f, destination)
        logger.info(f"Moved {f} to {destination}")


def main(default_config=False):
    if default_config:
        params = spk_config.set_config(default=True)
    else:
        params = spk_config.set_config()

    # If the script is being run automatically, on Fridays it will run a greater number of files
    if params["run_type"] == "Auto":
        if datetime.datetime.weekday(datetime.date.today()) == 4:
            n_files = int(params["weekend_run"])
        else:
            n_files = int(params["weekday_run"])
    elif params["run_type"] == "Manual":
        n_files = params["manual_run"]
    else:
        raise Exception('Run type choice is not valid. Options are "Manual" or "Auto"')

    runpath = Path(params["run_path"])
    runfiles = [f for f in runpath.glob("*.hdf5")][:n_files]

    num_cpu = int(params["cores_used"])
    resort_limit = int(params["resort_limit"])

    for curr_file in runfiles:  # loop through each file

        # Create the necessary directories
        dir_manager = DirectoryManager(curr_file)
        dir_manager.flush_directories()
        dir_manager.create_base_directories()

        h5file = spk_io.h5.read_h5(curr_file)
        num_chan = len(h5file.keys())
        dir_manager.create_channel_directories(num_chan)

        runs = math.ceil(num_chan / num_cpu)
        for n in range(runs):
            channels_per_run = (
                num_chan // runs
            )
            chan_start = n * channels_per_run
            chan_end = (n + 1) * channels_per_run if n < (runs - 1) else num_chan
            if chan_end > num_chan:
                chan_end = num_chan

            processes = []
            for i in range(chan_start, chan_end):
                this_chan = h5file[list(h5file.keys())[i]]
                dir_manager.idx = i
                p = multiprocessing.Process(
                    target=run_spk_process, args=(this_chan, dir_manager, i, params)
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

if __name__ == "__main__":
    main(default_config=True)
