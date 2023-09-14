from __future__ import annotations
import configparser
from pathlib import Path


def set_config(path: Path | str = '', default=False) -> dict[str, str | int | float]:
    if default:
        path = Path().home() / 'spk2py' / 'autosort' / 'default_config.ini'

    if not path.is_file():
        default_config(path)
        print(f'Default configuration file has been created. You can find it in {path}')
        return read_config(path)
    else:
        return read_config(path)


def default_config(path: Path, config_ver: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    default_data_path = Path().home() / 'data'
    default_run_path = default_data_path / 'to_run'
    default_results_path = default_data_path / 'results'
    default_completed_path = default_data_path / 'completed'

    config = configparser.ConfigParser()
    config['run-settings'] = {
        'resort-limit': '3',
        'cores-used': '8',
        'weekday-run': '2',
        'weekend-run': '8',
        'run-type': 'Auto',
        'manual-run': '2'
    }
    config['paths'] = {
        'run-path': str(default_run_path),
        'results-path': str(default_results_path),
        'completed-path': str(default_completed_path),
    }
    config['clustering'] = {
        'max-clusters': '7',
        'max-iterations': '1000',
        'convergence-criterion': '.0001',
        'random-restarts': '10',
        'l-ratio-cutoff': '.1'
    }
    config['signal'] = {
        'disconnect-voltage': '1500',
        'max-breach-rate': '.2',
        'max-breach-count': '10',
        'max-breach-avg': '20',
        'intra-cluster-cutoff': '3'
    }
    config['filtering'] = {
        'low-cutoff': '600',
        'high-cutoff': '3000'
    }
    config['spike'] = {
        'pre-time': '.2',
        'post-time': '.6',
        'sampling-rate': '20000'
    }
    config['std-dev'] = {
        'spike-detection': '2.0',
        'artifact-removal': '10.0'
    }
    config['pca'] = {
        'variance-explained': '.95',
        'use-percent-variance': '1',
        'principal-component-n': '5'
    }
    config['post-process'] = {
        'reanalyze': '0',
        'simple-gmm': '1',
        'image-size': '70',
        'temporary-dir': str(Path.home() / 'tmp_python')
    }
    config['version'] = {
        'config-version': str(config_ver)
    }

    with open(path, 'w') as configfile:
        config.write(configfile)


def read_config(path: Path, config_ver: int = 5):
    config = configparser.ConfigParser()
    config.read(path)

    if 'config-version' not in config['version'] or int(config['version']['config-version']) != config_ver:
        new_path = path.with_suffix(f'.{config["version"].get("config-version", "unknown")}.txt')
        path.rename(new_path)
        default_config(path, config_ver)
        print(
            f'Config version updated, config file reset to default, your original config file has been renamed.'
            f' Find the new config file here: {path}'
        )
        config = configparser.ConfigParser()
        config.read(path)

    return get_config_params(config)


def get_config_params(config: configparser.ConfigParser) -> dict[str, str | int | float]:
    return {
        "resort_limit": config.getint('run-settings', 'resort-limit'),
        "cores_used": config.getint('run-settings', 'cores-used'),
        "weekday_run": config.getint('run-settings', 'weekday-run'),
        "weekend_run": config.getint('run-settings', 'weekend-run'),
        "run_type": config.get('run-settings', 'run-type'),
        "manual_run": config.getint('run-settings', 'manual-run'),

        "to_run_path": config.get('paths', 'to-run-path'),
        "running_path": config.get('paths', 'running-path'),
        "results_path": config.get('paths', 'results-path'),
        "completed_path": config.get('paths', 'completed-path'),
        "use_path": config.getint('paths', 'use-path'),
        "else_path": config.get('paths', 'else-path'),

        "max_clusters": config.getint('clustering', 'max-clusters'),
        "max_iterations": config.getint('clustering', 'max-iterations'),
        "convergence_criterion": config.getfloat('clustering', 'convergence-criterion'),
        "random_restarts": config.getint('clustering', 'random-restarts'),
        "l_ratio_cutoff": config.getfloat('clustering', 'l-ratio-cutoff'),

        "disconnect_voltage": config.getint('signal', 'disconnect-voltage'),
        "max_breach_rate": config.getfloat('signal', 'max-breach-rate'),
        "max_breach_count": config.getint('signal', 'max-breach-count'),
        "max_breach_avg": config.getint('signal', 'max-breach-avg'),
        "intra_cluster_cutoff": config.getint('signal', 'intra-cluster-cutoff'),

        "low_cutoff": config.getint('filtering', 'low-cutoff'),
        "high_cutoff": config.getint('filtering', 'high-cutoff'),

        "pre_time": config.getfloat('spike', 'pre-time'),
        "post_time": config.getfloat('spike', 'post-time'),
        "sampling_rate": config.getint('spike', 'sampling-rate'),

        "spike_detection": config.getfloat('std-dev', 'spike-detection'),
        "artifact_removal": config.getfloat('std-dev', 'artifact-removal'),

        "variance_explained": config.getfloat('pca', 'variance-explained'),
        "use_percent_variance": config.getint('pca', 'use-percent-variance'),
        "principal_component_n": config.getint('pca', 'principal-component-n'),

        "reanalyze": config.getint('post-process', 'reanalyze'),
        "simple_gmm": config.getint('post-process', 'simple-gmm'),
        "image_size": config.getint('post-process', 'image-size'),
        "temporary_dir": config.get('post-process', 'temporary-dir'),

        "config_version": config.get('version', 'config-version'),
    }
