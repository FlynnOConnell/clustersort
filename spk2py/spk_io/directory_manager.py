from pathlib import Path


class DirectoryManager:
    def __init__(self, base_path):
        self.base_path = Path(base_path)

    @property
    def raw_data(self):
        return self.base_path / "Raw Data"

    @property
    def processed_data(self):
        return self.base_path / "Processed Data"

    @property
    def plots(self):
        return self.base_path / "Plots"

    @property
    def reports(self):
        return self.base_path / "Reports"

    @property
    def temporary(self):
        return self.base_path / "Temporary"

    def create_base_directories(self):
        """Creates the base directories for raw data, processed data, plots, reports, and temporary files."""
        directories = [
            self.raw_data,
            self.processed_data,
            self.plots,
            self.reports,
            self.temporary,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def create_channel_directories(self, channel_number, sub_dirs):
        """Creates a set of subdirectories for a specific channel under each base directory."""
        base_dirs = [self.raw_data, self.processed_data, self.plots, self.reports]
        for base_dir in base_dirs:
            channel_dir = base_dir / f"channel_{channel_number}"
            channel_dir.mkdir(parents=True, exist_ok=True)
            for sub_dir in sub_dirs:
                (channel_dir / sub_dir).mkdir(parents=True, exist_ok=True)
