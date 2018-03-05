from noodles.run.xenon import (
    Machine, XenonJobConfig, run_xenon)
from noodles import gather_all, schedule
from noodles.tutorial import (add, sub, mul)
import xenon
from pathlib import Path
import configparser
import sys


def test_xenon_42_multi(run):
    A = add(1, 1)
    B = sub(3, A)

    multiples = [mul(add(i, B), A) for i in range(6)]
    C = schedule(sum)(gather_all(multiples))

    result = run(C)
    print("The answer is:", result)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read('mcfly.ini')

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
        print("The mcfly.ini file should provide a certificate or "
              "a password.")
        sys.exit(1)

    print(dict(config['xenon']))
    print(dict(config['xenon.properties']))
    print(dict(config['xenon.job_options']))

    xenon.init()
    machine = Machine(
        scheduler_adaptor=config['xenon']['adaptor'],
        location=config['xenon']['location'],
        credential=credential,
        jobs_properties=dict(config['xenon.properties']))

    worker_config = XenonJobConfig(
        prefix=Path(config['job_config']['python_prefix']),
        working_dir=config['job_config']['working_directory'],
        time_out=config.getint('xenon', 'timeout', fallback=0),
        options=dict(config['xenon.job_options']),
        verbose=False)  # , options=['-C', 'TitanX', '--gres=gpu:1'])

    def run(workflow):
        return run_xenon(
            workflow,
            machine=machine,
            worker_config=worker_config,
            n_processes=2)

    test_xenon_42_multi(run)
