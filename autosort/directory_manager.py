"""
=================
Directory Manager
=================

Class for managing directories produced and utilized when running the autosort pipeline.

"""

from __future__ import annotations
import logging
import shutil
from pathlib import Path


class DirectoryManager:
    """
    Manages directories for storing processed data, plots, reports, and intermediate files.

    Attributes
    ----------
    filename : str
        Stem of the input filepath.
    base_path : Path
        Parent directory for the main categories of directories.
    base_suffix : str
        Suffix of the base path.
    directories : list of Path
        List of all primary directories managed by this instance.
    idx : int
        Index used for channel directory creation, default is 0.
    """

    def __init__(self, filepath: str | Path):
        """
        Initialize DirectoryManager.

        Parameters
        ----------
        filepath : str or Path
            Full filepath of the data file being processed.
        """
        self.filename = Path(filepath).stem
        self.base_path = Path(filepath).parent / self.filename
        self.base_suffix = self.base_path.suffix
        self.directories = [
            self.processed,
            self.plots,
            self.reports,
            self.intermediate,
        ]
        self.idx = 0
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())

    @property
    def processed(self):
        """
        Path : Directory for storing processed data.
        """
        return self.base_path / "Processed"

    @property
    def plots(self):
        """
        Path : Directory for storing plots.
        """
        return self.base_path / "Plots"

    @property
    def reports(self):
        """
        Path : Directory for storing reports.
        """
        return self.base_path / "Reports"

    @property
    def intermediate(self):
        """
        Path : Directory for storing intermediate files.
        """
        return self.base_path / "Intermediate"

    @property
    def channel(self):
        """
        int : Channel index incremented by 1.
        """
        return self.idx + 1

    def create_base_directories(self):
        """
        Creates the base directories for raw data, processed data, plots, reports, and temporary files.
        """
        for directory in self.directories:
            directory.mkdir(parents=True, exist_ok=True)

    def flush_directories(self):
        """
        Deletes all files and subdirectories in each base directory.

        Raises
        ------
        Exception
            If there is an error in deleting files or directories.
        """
        try:
            for base_dir in self.directories:
                for f in base_dir.glob("*"):
                    self.logger.debug(f"Found base_dir: {f}")
                    if f.is_file():
                        self.logger.debug(f"Deleting file: {f}")
                        f.unlink()
                    elif f.is_dir():
                        self.logger.debug(f"Deleting directory: {f}")
                        shutil.rmtree(f)
        except Exception as e:
            self.logger.error(f"Error flushing directories: {e}", exc_info=True)

    def create_channel_directories(self, num_chan):
        """
        Creates a set of subdirectories for a specific channel under each base directory.

        Parameters
        ----------
        num_chan : int
            Number of channels for which to create directories.
        """
        base_dirs = [self.processed, self.plots, self.reports, self.intermediate]
        for base_dir in base_dirs:
            for channel_number in range(1, num_chan + 1):
                channel_dir = base_dir / f"channel_{channel_number}"
                self.logger.debug(f"Creating channel directory: {channel_dir}")
                channel_dir.mkdir(parents=True, exist_ok=True)
