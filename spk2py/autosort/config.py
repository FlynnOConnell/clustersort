import os
import configparser
from pathlib import Path

config_ver = 5


def do_the_config(path=''):
    path = Path().home() / 'autosort' / 'autosort_config.ini' if not path else Path(path)
    path = Path(os.path.expanduser('~')) / 'autosort' / 'autosort_config.ini'

    if not path.is_file():
        default_config(path)
        print(f'Default configuration file has been created. You can find it in {path}')
    else:
        return read_config(path)


def default_config(path):
    path.parent.mkdir(parents=True, exist_ok=True)

    config = configparser.ConfigParser()
    config['Run Settings'] = {
        'Resort Limit': '3',
        'Cores Used': '8',
        'Weekday Run': '2',
        'Weekend Run': '8',
        'Run Type': 'Auto',
        'Manual Run': '2'
    }
    config['Paths'] = {
        'to-run-path': '',
        'running-path': '',
        'results-path': '',
        'completed-path': '',
        'use-path': '1',
        'Else Path': ''
    }
    config['Clustering'] = {
        'Max Clusters': '7',
        'Max Iterations': '1000',
        'Convergence Criterion': '.0001',
        'Random Restarts': '10',
        'L-ratio Cutoff': '.1'
    }
    config['Signal'] = {
        'Disconnect Voltage': '1500',
        'Max Breach Rate': '.2',
        'Max Breach Count': '10',
        'Max Breach Avg.': '20',
        'Intra-Cluster Cutoff': '3'
    }
    config['Filtering'] = {
        'Low Cutoff': '600',
        'High Cutoff': '3000'
    }
    config['Spike'] = {
        'Pre-time': '.2',
        'Post-time': '.6',
        'Sampling Rate': '20000'
    }
    config['Std Dev'] = {
        'Spike Detection': '2.0',
        'Artifact Removal': '10.0'
    }
    config['PCA'] = {
        'Variance Explained': '.95',
        'Use Percent Variance': '1',
        'Principal Component n': '5'
    }
    config['Post Process'] = {
        'reanalyze': '0',
        'simple gmm': '1',
        'image size': '70',
        'temporary dir': Path().home() / 'tmp_python'
    }
    config['Version'] = {
        'config version': str(config_ver)
    }

    with open(path, 'w') as configfile:
        config.write(configfile)


def read_config(path):
    config = configparser.ConfigParser()
    params = {}
    config.read(path)
    for key, value in config._sections.items():
        params.update(value)

    if config_ver != int(params['config version']):
        path.rename(path.with_suffix(str(params['config version']) + '.txt'))
        default_config(path)
        print(
            f'Config version updated, config file reset to default, your original config file has been renamed.'
            f' Find the new config file here: {path}'
        )
    else:
        return params
