from noodles.run.xenon import (
    Machine, XenonJobConfig, run_xenon)
import xenon
from pathlib import Path
import configparser
import sys
import shlex

from .storage import serial_registry


def run_remote(workflow, config_file='mcfly.ini'):
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_file)

    if 'certificate' in config['xenon']:
        credential = xenon.CertificateCredential(
            username=config['xenon']['user'],
            certfile=config['xenon']['certificate'])
        if 'passphrase' in config['xenon']:
            credential.passphrase = config['xenon']['passphrase']
    elif 'password' in config['xenon']:
        credential = xenon.PasswordCredential(
            username=config['xenon']['user'],
            password=config['xenon']['password'])
    else:
        credential = None

    xenon.init()
    machine = Machine(
        scheduler_adaptor=config['xenon']['adaptor'],
        location=config['xenon']['location'],
        credential=credential,
        jobs_properties=dict(config['xenon.properties']))

    worker_config = XenonJobConfig(
        prefix=Path(config['job_config']['python_prefix']),
        working_dir=config['job_config']['working_directory'],
        registry=serial_registry,
        time_out=config.getint('xenon', 'timeout', fallback=0),
        scheduler_arguments=shlex.split(
            config['job_config']['scheduler_arguments']),
        verbose=True)

    return run_xenon(
        workflow,
        machine=machine,
        worker_config=worker_config,
        n_processes=int(config['xenon']['n_nodes']))

