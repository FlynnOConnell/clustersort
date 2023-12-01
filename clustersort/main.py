# -*- coding: utf-8 -*-
"""

"""
from __future__ import annotations

import datetime
import math
import multiprocessing
from pathlib import Path

import h5py

from clustersort.directory_manager import DirectoryManager
from clustersort.logger import logger
from clustersort.sort import sort
from clustersort.spk_config import SortConfig
from clustersort.utils.progress import ProgressBarManager

def __read_group(group: h5py.Group) -> dict:
    data = {}
    for attr_name, attr_value in group.attrs.items():
        data[attr_name] = attr_value
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            data[key] = __read_group(item)
        elif isinstance(item, h5py.Dataset):
            data[key] = item[()]
    return data


def read_h5(filename: str | Path) -> dict:
    with h5py.File(filename, "r") as f:
        data = __read_group(f)
    return data


def check_substring_content(main_string, substring):
    """Checks if any combination of the substring is in the main string."""
    return substring.lower() in main_string.lower()

def run(params: SortConfig, parallel: bool = True, overwrite: bool = False):
    """
    Entry point for the clustersort package.
    Optionally include a `SpkConfig` object to override the default parameters.

    This function iterates over data files, manages directories, and executes sorting either sequentially
    or in parallel.

    Parameters
    ----------
    params : spk_config.SortConfig
        Configuration parameters for spike sorting. If `None`, default parameters are used.
    parallel : bool, optional
        Whether to run the sorting in parallel. Default is `True`.
    overwrite : bool, optional
        Whether to overwrite existing files. Default is `False`.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If the run type specified in `params` is not either "Manual" or "Auto".

    Examples
    --------
    >>> sort(SortConfig(), parallel=True)
    """

    pbm = ProgressBarManager()
    if not params:
        logger.info("No parameters provided. Using default parameters.")
        params = SortConfig()
    else:
        logger.info("Using provided parameters:")
        params = params
        logger.info(f"{params.get_section('path')}")
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

    runpath = Path(params.path["data"])
    num_cpu = int(params.run["cores-used"]) if parallel else 1
    runfiles = [f for f in runpath.iterdir() if f.is_file()][:n_files]
    pbm.init_file_bar(len(runfiles))

    for curr_file in runfiles:
        logger.info(f"Processing file: {curr_file}")
        h5file = read_h5(curr_file)
        all_data = h5file["data"]

        unit_data = {}
        for key in all_data.keys():
            if check_substring_content(key, "U"):
                unit_data[key] = all_data[key]
        num_chan = len(unit_data)

        # Create the necessary directories
        dir_manager = DirectoryManager(
            curr_file,
            num_chan,
            params,
        )
        dir_manager.flush_directories()
        dir_manager.create_base_directories()
        dir_manager.create_channel_directories()

        runs = math.ceil(num_chan / num_cpu)
        pbm.init_channel_bar(runs)
        for n in range(runs):
            channels_per_run = num_chan // runs
            chan_start = n * channels_per_run
            chan_end = (n + 1) * channels_per_run if n < (runs - 1) else num_chan
            if chan_end > num_chan:
                chan_end = num_chan

            if parallel:
                processes = []
                for i in range(chan_start, chan_end):
                    chan_name = [list(h5file["data"].keys())[i]][0]
                    chan_data = h5file["data"][chan_name]
                    sampling_rate = h5file["metadata_channel"][chan_name][
                        "sampling_rate"
                    ]
                    dir_manager.idx = i
                    p = multiprocessing.Process(
                        target=sort,
                        args=(
                            curr_file,
                            chan_data,
                            sampling_rate,
                            params,
                            dir_manager,
                            i,
                        ),
                    )
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
            else:
                for i in range(chan_start, chan_end):
                    chan_name = [list(unit_data.keys())[i]][0]
                    chan_data = unit_data[chan_name]
                    sampling_rate = h5file["metadata_channel"][chan_name][
                        "fs"
                    ]
                    dir_manager.idx = i
                    sort(
                        curr_file,
                        chan_data,
                        sampling_rate,
                        params,
                        dir_manager,
                        i,
                        overwrite=overwrite,
                    )
            pbm.update_channel_bar()
        pbm.update_file_bar()
    pbm.close_channel_bar()
    pbm.close_file_bar()


if __name__ == "__main__":
    logger.setLevel("DEBUG")
    main_params = SortConfig()
    my_data = Path.home() / "spk2extract" / "combined"
    main_params.set("path", "data", my_data)
    main_params.save_to_ini()
    run(main_params, parallel=False, overwrite=False)
