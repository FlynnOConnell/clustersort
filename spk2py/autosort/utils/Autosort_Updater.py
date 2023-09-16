# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 09:58:45 2020

@author: Di Lorenzo Tech
"""
import os
from pathlib import Path
import site
from configparser import ConfigParser
import shutil
import sys
from datetime import date
from spk2py.autosort import config

python_path = Path(site.getsitepackages()[0])
autosort_repo = Path().home() / 'repos' / 'spk2py' / 'autosort'
autosort_path = Path().home() / 'autosort'


def update_autosort():
    # check that the python path is valid
    if not os.path.isdir(python_path):
        sys.exit('Your python path is incorrect')

    # replace pypl2 directory
    if os.path.isdir(python_path / '/pypl2'):
        shutil.rmtree(python_path / '/pypl2')
    shutil.copytree(autosort_repo / '/pypl2', python_path / '/pypl2')

    # Make the autosort folder if non exists
    if not os.path.isdir(autosort_path):
        os.mkdir(autosort_path)
        shutil.copy(autosort_repo / '/Json2Nex.py', autosort_path)
        shutil.copy(autosort_repo / '/utils/Autosort_Updater.py', autosort_path)
    shutil.copy(autosort_repo / '/main.py', autosort_path)
    shutil.copy(autosort_repo / '/autosort_post.py', autosort_path)

    # make config file
    config.set_config(autosort_path / '/Autosort_config.ini')


if os.path.isfile(autosort_path / '/version.info'):
    config = ConfigParser()
    config.read(autosort_repo / '/version.txt')
    version = config['Autosort']['version']
    config.read(autosort_path / '/version.info')
    oldver = config['Autosort']['version']
    if oldver != version:
        print('Your Autosort is outdated. Updating...')
        update_autosort()
        config['Autosort'] = {'version': version, 'last update': str(date.today())}
        with open(autosort_path / '/version.info', 'w') as outfile:
            config.write(outfile)
        print('Update complete! Version number is ' + version)
    else:
        print('Your Autosort is up to date! Version number is ' + version)
else:
    print('Installing Autosort...')
    update_autosort()
    config = ConfigParser()
    config.read(autosort_repo / '/version.txt')
    version = config['Autosort']['version']
    config['Autosort'] = {'version': version, 'last update': str(date.today())}
    with open(autosort_path / '/version.info', 'w') as outfile:
        config.write(outfile)
    print('Installation complete! Version number is ' + version)
    print(
        'It is recommended that you create a batch file to activate this script weekly using Windows task scheduler, '
        'in order to avoid missing important updates!\nIf you plan on running the automated version of the Autosort, '
        'you will need to do the same for the Autosort_main.py script.')
