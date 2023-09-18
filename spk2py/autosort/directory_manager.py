import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logpath = Path().home() / 'autosort' / "directory_logs.log"
logging.basicConfig(filename=logpath, level=logging.DEBUG)
logger.addHandler(logging.StreamHandler())

class DirectoryManager:
    def __init__(self, filepath):
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

    @property
    def processed(self,):
        return self.base_path / "Processed"

    @property
    def plots(self,):
        return self.base_path / "Plots"

    @property
    def reports(self,):
        return self.base_path / "Reports"

    @property
    def intermediate(self,):
        return self.base_path / "Intermediate"

    @property
    def channel(self,):
        return self.idx + 1

    def create_base_directories(self,):
        """Creates the base directories for raw data, processed data, plots, reports, and temporary files."""
        for directory in self.directories:
            directory.mkdir(parents=True, exist_ok=True)

    def flush_directories(self,):
        """Deletes all files and subdirectories in each base directory."""
        try:
            for base_dir in self.directories:
                for f in base_dir.glob("*"):
                    logger.debug(f"Found base_dir: {f}")
                    if f.is_file():
                        logger.debug(f"Deleting file: {f}")
                        f.unlink()
                    elif f.is_dir():
                        logger.debug(f"Deleting directory: {f}")
                        shutil.rmtree(f)
        except Exception as e:
            logger.error(f"Error flushing directories: {e}", exc_info=True)

    def create_channel_directories(self, num_chan):
        """Creates a set of subdirectories for a specific channel under each base directory."""
        base_dirs = [self.processed, self.plots, self.reports, self.intermediate]
        for base_dir in base_dirs:
            for channel_number in range(1, num_chan + 1):
                channel_dir = base_dir / f"channel_{channel_number}"
                logger.debug(f"Creating channel directory: {channel_dir}")
                channel_dir.mkdir(parents=True, exist_ok=True)
