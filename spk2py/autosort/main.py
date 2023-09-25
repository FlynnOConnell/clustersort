# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime
import logging
import math
import multiprocessing
import shutil
from pathlib import Path

import spk_config
from spk2py.spk_io import read_h5
from autosort import run_spk_process
from directory_manager import DirectoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(params: spk_config.SpkConfig, parallel: bool = True):
    """
    Entry point for the autosort package.
    Optionally include a `SpkConfig` object to override the default parameters.

    This function iterates over data files, manages directories, and executes sorting either sequentially
    or in parallel.

    Parameters
    ----------
    params : spk_config.SpkConfig
        Configuration parameters for spike sorting. If `None`, default parameters are used.
    parallel : bool, optional
        Whether to run the sorting in parallel. Default is `True`.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If the run type specified in `params` is not either "Manual" or "Auto".

    Examples
    --------
    >>> main(spk_config.SpkConfig(), parallel=True)
    """
    if not params:
        params = spk_config.SpkConfig()
    else:
        params = params
    # If the script is being run automatically, on Fridays it will run a greater number of files
    if params.run["run-type"] == "Auto":
        if datetime.datetime.weekday(datetime.date.today()) == 4:
            n_files = int(params.run["weekend-run"])
        else:
            n_files = int(params.run["weekday-run"])
    elif params.run["run-type"] == "Manual":
        n_files = params.run["manual-run"]
    else:
        raise Exception('Run type choice is not valid. Options are "Manual" or "Auto"')

    runpath = Path(params.path["run"])
    num_cpu = int(params.run["cores-used"]) if parallel else 1
    runfiles = [f for f in runpath.iterdir() if f.is_file()][:n_files]
    for curr_file in runfiles:  # loop through each file

        # Create the necessary directories
        dir_manager = DirectoryManager(curr_file)
        dir_manager.flush_directories()
        dir_manager.create_base_directories()

        h5file = read_h5(curr_file)
        unit_data = h5file["unit"]
        num_chan = len(unit_data)
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

            if parallel:
                processes = []
                for i in range(chan_start, chan_end):
                    chan_name = [list(h5file['unit'].keys())[i]][0]
                    chan_data = h5file['unit'][chan_name]
                    dir_manager.idx = i
                    p = multiprocessing.Process(
                        target=run_spk_process, args=(curr_file, chan_data, params, dir_manager, i)
                    )
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
            else:
                for i in range(chan_start, chan_end):
                    chan_name = [list(h5file['unit'].keys())[i]][0]
                    chan_data = h5file['unit'][chan_name]
                    dir_manager.idx = i
                    run_spk_process(curr_file, chan_data, params, dir_manager, i)


if __name__ == "__main__":
    main_params = spk_config.SpkConfig()
    main_params.set("path", "run", Path.home() / "data" / "combined")
    main(main_params, parallel=False)
