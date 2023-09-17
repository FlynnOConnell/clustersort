from pathlib import Path
import shutil


class DirectoryManager:
    def __init__(self, filepath):
        self.base_path = Path(filepath).parent / Path(filepath).stem
        self.base_suffix = self.base_path.suffix
        self.directories = [
            self.raw_data,
            self.processed_data,
            self.plots,
            self.reports,
            self.temporary,
        ]
        self.idx = 0

    @property
    def raw_data(self):
        return self.base_path / "Raw"

    @property
    def processed_data(self):
        return self.base_path / "Processed"

    @property
    def plots(self):
        return self.base_path / "Plots"

    @property
    def reports(self):
        return self.base_path / "Reports"

    @property
    def temporary(self):
        return self.base_path / "temp"

    @property
    def channel(self):
        return self.idx + 1

    def create_base_directories(self):
        """Creates the base directories for raw data, processed data, plots, reports, and temporary files."""
        for directory in self.directories:
            directory.mkdir(parents=True, exist_ok=True)

    def flush_directories(self):
     """Deletes all files and subdirectories in each base directory."""
     for base_dir in self.directories:
         if base_dir.exists():
             for item in base_dir.iterdir():
                 if item.is_dir():
                     shutil.rmtree(item)
                 else:
                     item.unlink()

    def create_channel_directories(self, channel_number, sub_dirs):
        """Creates a set of subdirectories for a specific channel under each base directory."""
        base_dirs = [self.raw_data, self.processed_data, self.plots, self.reports, self.temporary]
        for base_dir in base_dirs:
            channel_dir = base_dir / f"channel_{channel_number}"
            channel_dir.mkdir(parents=True, exist_ok=True)
            for sub_dir in sub_dirs:
                (channel_dir / sub_dir).mkdir(parents=True, exist_ok=True)
