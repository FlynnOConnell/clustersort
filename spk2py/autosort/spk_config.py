"""
.. module:: spk_config
    :synopsis: Module to manage configurations for the AutoSort pipeline.

.. moduleauthor:: Flynn O'Connell <


"""
from __future__ import annotations

from typing import Any
import configparser
from pathlib import Path


class SpkConfig:
    """
    Class to manage configurations for the AutoSort pipeline.

    This class reads from an INI-style configuration file and provides methods
    to access the configurations for different sections. These sections include
    'run', 'path', 'cluster', and so on.

    Parameters
    ----------
    cfg_path : str or Path, optional
        The path to the configuration file. Defaults to a pre-defined location.

    Notes
    -----
    The configuration file is an INI-style file with the following sections:

    - run : dict
        Contains configurations related to runtime settings like 'resort-limit', 'cores-used'.
    - path : dict
        Contains path settings like directories for 'run', 'results'.
    - cluster : dict
        Contains clustering parameters like 'max-clusters', 'max-iterations'.
    - breach : dict
        Contains breach analysis parameters like 'disconnect-voltage', 'max-breach-rate'.
    - filter : dict
        Contains filter parameters like 'low-cutoff', 'high-cutoff'.
    - spike : dict
        Contains spike-extraction settings like 'pre-time', 'post-time'.

    See Also: `configparser from python std library <https://docs.python.org/3/library/configparser.html>`_

    """
    def __init__(self, cfg_path: Path | str = ""):
        """
        Initialize a new SpkConfig object to manage configurations for the AutoSort pipeline.

        This class reads from an INI-style configuration file and provides methods
        to access the configurations for different sections. These sections include
        'run', 'path', 'cluster', and so on.

        Parameters
        ----------
        cfg_path : str or Path, optional
            The path to the configuration file. Defaults to a pre-defined location.

        Notes
        -----
        The configuration file is an INI-style file with the following sections:

        - run : dict
            Contains configurations related to runtime settings like 'resort-limit', 'cores-used'.
        - path : dict
            Contains path settings like directories for 'run', 'results'.
        - cluster : dict
            Contains clustering parameters like 'max-clusters', 'max-iterations'.
        - breach : dict
            Contains breach analysis parameters like 'disconnect-voltage', 'max-breach-rate'.
        - filter : dict
            Contains filter parameters like 'low-cutoff', 'high-cutoff'.
        - spike : dict
            Contains spike-related settings like 'pre-time', 'post-time'.

        .. note::
           Due to the nature of INI files, all values are stored as strings. It is up to the user
           to convert the values to the appropriate type.

        Examples
        --------
        >>> cfg = SpkConfig()
        >>> run = cfg.run
        >>> print(type(run), run)
        <class 'dict'> {'resort-limit': '3', 'cores-used': '8', ...}

        >>> cfg.set('run', 'resort-limit', 5)
        >>> print(cfg.run['resort-limit'])
        '5'
        """
        if not cfg_path:
            self.cfg_path = Path().home() / "autosort" / "autosort_config.ini"
            self.set_default_config()
        else:
            self.cfg_path = cfg_path
        self.config = self.read_config()
        self.params = self.get_all()
        self._validate_config()

    def get_section(self, section: str):
        """
            Returns a dictionary containing key-value pairs for the given section.

            Parameters
            ----------
            section : str
                The name of the section in the config file.

            Returns
            -------
            dict
                Dictionary containing the section's key-value pairs.
            """
        return dict(self.config[section])

    def set(self, section: str, key: str, value: Any):
        """
        Set the value of a configuraton parameter.

        Parameters
        ----------
        section : str
        key : str
        value : Any

        Returns
        -------
        None
        """

        if section not in self.config:
            self.config.add_section(section)
        self.config.set(section, key, str(value))

    def get_all(self):
        """
        Returns
        -------
        dict
            A dictionary containing all key-value pairs from all sections.
        """
        params = {}
        for section in self.config.sections():
            for key, value in self.config.items(section):
                params[key] = value
        return params

    @property
    def run(self):
        """
            Returns
            -------
            dict
                A dictionary containing key-value pairs for the 'run' section.
            """
        return self.get_section('run')

    @property
    def path(self):
        """
            Returns
            -------
            dict
                A dictionary containing key-value pairs for the 'path' section.
            """
        return self.get_section('path')

    @property
    def cluster(self):
        """
            Returns
            -------
            dict
                A dictionary containing key-value pairs for the 'cluster' section.
            """
        return self.get_section('cluster')

    @property
    def breach(self):
        """
            Returns
            -------
            dict
                A dictionary containing key-value pairs for the 'breach' section.
            """
        return self.get_section('breach')

    @property
    def filter(self):
        """
            Returns
            -------
            dict
                A dictionary containing key-value pairs for the 'breach' section.
            """
        return self.get_section('filter')

    @property
    def spike(self):
        """
           Returns
           -------
           dict
               A dictionary containing key-value pairs for the 'spike' section.
           """
        return self.get_section('spike')

    @property
    def detection(self):
        """
            Returns
            -------
            dict
                A dictionary containing key-value pairs for the 'detection' section.
            """
        return self.get_section('detection')

    @property
    def pca(self):
        """
            Returns
            -------
            dict
                A dictionary containing key-value pairs for the 'pca' section.
            """
        return self.get_section('pca')

    @property
    def postprocess(self):
        """
            Returns
            -------
            dict
                A dictionary containing key-value pairs for the 'postprocess' section.
            """
        return self.get_section('postprocess')

    def read_config(self):
        """
            Returns
            -------
            ConfigParser
                A ConfigParser object loaded with the INI file.
            """
        config = configparser.ConfigParser()
        config.read(self.cfg_path)
        return config

    def set_config(self, default=False):
        """
            Parameters
            ----------
            default : bool, optional
                If True, sets the configuration to its default settings. Defaults to False.
            """
        if default:
            self.set_default_config()
    
        if not self.cfg_path.is_file() or default:
            self.set_default_config()
            print(f"Default configuration file has been created. You can find it in {self.cfg_path}")

    def set_default_config(self) -> None:
        """
            Sets the default configurations for all sections. Writes these to the configuration file.
            """
        assert self.cfg_path.parent.is_dir(), f"Parent directory {self.cfg_path.parent} does not exist"
        self.cfg_path.parent.mkdir(parents=True, exist_ok=True)
        default_data_path = self.cfg_path.parent
        default_run_path = default_data_path / "to_run"
        default_run_path.mkdir(parents=True, exist_ok=True)
        default_results_path = default_data_path / "results"
        default_results_path.mkdir(parents=True, exist_ok=True)
        default_completed_path = default_data_path / "completed"
        default_completed_path.mkdir(parents=True, exist_ok=True)
    
        config = configparser.ConfigParser()
        config["run"] = {
            "resort-limit": "3",
            "cores-used": "8",
            "weekday-run": "2",
            "weekend-run": "8",
            "run-type": "Auto",
            "manual-run": "2",
        }
    
        config["path"] = {
            "run": str(default_run_path),  # Path to directory containing files to be sorted
            "results": str(default_results_path),  # Path to directory to store results
            "completed": str(default_completed_path),  # Path to directory to store completed files
        }
    
        config["cluster"] = {
            "max-clusters": "7",
            "max-iterations": "1000",
            "convergence-criterion": ".0001",
            "random-restarts": "10",
            "l-ratio-cutoff": ".1",
            "intra-cluster-cutoff": "3",
        }
    
        config["breach"] = {
            "disconnect-voltage": "1500",
            "max-breach-rate": ".2",
            "max-breach-count": "10",
            "max-breach-avg": "20",
        }
    
        config["filter"] = {"low-cutoff": "300", "high-cutoff": "3000"}
    
        config["spike"] = {"pre-time": "0.2", "post-time": "0.6",}
    
        config["detection"] = {"spike-detection": "2.0", "artifact-removal": "10.0"}
    
        config["pca"] = {
            "variance-explained": ".95",
            "use-percent-variance": "1",
            "principal-component-n": "5",
        }
    
        config["postprocess"] = {
            "reanalyze": "0",
            "simple-gmm": "1",
            "image-size": "70",
            "temporary-dir": str(Path.home() / "tmp_python"),
        }

        with open(self.cfg_path, "w") as configfile:
            config.write(configfile)

    def _validate_config(self):
        """
            Validates the loaded configurations to ensure they meet specified criteria.

            Raises
            ------
            AssertionError
                If any of the loaded configurations are not valid.
            """
        assert(self.cfg_path.is_file()), f"Configuration file {self.cfg_path} does not exist"
        assert(self.run["run-type"] in ["Auto", "Manual"]), (f"Run type {self.run['run-type']} is not valid. Options "
                                                             f"are 'Auto' or 'Manual'")


if __name__ == '__main__':
    test_cfg = SpkConfig()
    x = 5