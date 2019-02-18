import argparse
import glob
import logging
import math
import os
import shutil
import stat
import subprocess
import sys

from ..conda import clone_eqip_environment, create_eqip_environment, default_revisions

_CREATE_SETUP_TEMPLATE = """#!/usr/bin/env python3

import os

from eqip.experiment.affinities_with_glia import _create_setup

here = os.path.abspath(os.path.dirname(__file__))
_create_setup(experiment_dir=here)
"""

def _create_setup(experiment_dir):

    from .templates import make_architecture_no_docker, make_training_no_docker

    def _is_int(string):
        try:
            int(string)
            return True
        except ValueError:
            return False

    def bounded_integer(val, lower=None, upper=None):
        val = int(val)
        if lower is not None and val < lower or upper is not None and val > upper:
            raise argparse.ArgumentTypeError('Value %d is out of bounds for [%s, %s]' % (
            val, str(-math.inf if lower is None else lower), str(math.inf if upper is None else upper)))
        return val

    directories = tuple(int(os.path.basename(d)) for d in glob.glob(os.path.join(experiment_dir, '*')) if _is_int(os.path.basename(d)))
    max_setup_id = -1 if len(directories) == 0 else max(directories)
    setup_id = max_setup_id + 1

    while (True):
        try:
            os.makedirs(os.path.join(experiment_dir, str(setup_id)), exist_ok=False)
            break
        except:
            print('Setup', setup_id, 'already exists, trying next setup id')
            setup_id += 1

    setup_dir = os.path.join(experiment_dir, str(setup_id))


    parser = argparse.ArgumentParser()
    parser.add_argument('--affinity-neighborhood-x', required=True, nargs='+', type=int)
    parser.add_argument('--affinity-neighborhood-y', required=True, nargs='+', type=int)
    parser.add_argument('--affinity-neighborhood-z', required=True, nargs='+', type=int)
    parser.add_argument('--mse-iterations', required=True, type=lambda arg: bounded_integer(arg, lower=0))
    parser.add_argument('--malis-iterations', required=True, type=lambda arg: bounded_integer(arg, lower=0))
    parser.add_argument('--data-provider', required=False, default=os.path.join(experiment_dir, 'data/*'))
    parser.add_argument('--log-level', default='INFO', choices=('DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'))
    parser.add_argument('--additional-pip-packages', nargs='+', default=())

    args, unknown = parser.parse_known_args()
    logging.basicConfig(level=logging.getLevelName(args.log_level))

    if len(args.additional_pip_packages) == 0:
        os.symlink(os.path.join(os.pardir, 'conda-env'), os.path.join(setup_dir, 'conda-env'))
    else:
        clone_eqip_environment(
            os.path.join(setup_dir, 'conda-env'),
            os.path.join(experiment_dir, 'conda-env'),
            use_name_as_prefix=True,
            extra_pip_installs=args.additional_pip_packages)

    num_affinities = sum(len(n) for n in (
        args.affinity_neighborhood_x,
        args.affinity_neighborhood_y,
        args.affinity_neighborhood_z))

    with open(os.path.join(setup_dir, 'mknet.sh'), 'w') as f:
        f.write(make_architecture_no_docker(
            command='make-affinities-on-interpolated-ground-truth-with-glia',
            args='--num-affinities=%d' % num_affinities))

    ignore_args_parser = argparse.ArgumentParser()
    ignore_args_parser.add_argument('--docker-container', required=False)
    _, train_args = ignore_args_parser.parse_known_args()

    with open(os.path.join(setup_dir, 'train.sh'), 'w') as f:
        f.write(make_training_no_docker(
            command='train-affinities-on-interpolated-ground-truth-with-glia',
            args=' '.join(train_args)))

    os.chmod(os.path.join(setup_dir, 'mknet.sh'), stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP)
    os.chmod(os.path.join(setup_dir, 'train.sh'), stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP)

    print('Created setup %d for experiment %s' % (setup_id, experiment_dir))

def create_experiment(
        path,
        data_pattern,
        create_conda_env,
        symlink_data=False,
        overwrite=False):


    try:
        os.makedirs(path, exist_ok=False)
    except OSError as e:
        if overwrite:
            shutil.rmtree(path)
            return create_experiment(path=path, data_pattern=data_pattern, symlink_data=symlink_data, overwrite=overwrite, create_conda_env=create_conda_env)
        else:
            raise e

    data_dir      = os.path.join(path, 'data')
    conda_env_dir = os.path.join(path, 'conda-env')
    os.makedirs(data_dir, exist_ok=False)
    create_conda_env(conda_env_dir)
    with open(os.path.join(path, 'create-setup.py'), 'w') as f:
        f.write(_CREATE_SETUP_TEMPLATE)
        os.chmod(os.path.join(path, 'create-setup.py'), stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP)

    with open(os.path.join(path, 'data-source'), 'w') as f:
        f.write(data_pattern)

    for fn in glob.glob(data_pattern):
        base_name   = os.path.basename(fn)
        target_name = os.path.join(data_dir, base_name)
        if symlink_data:
            os.symlink(fn, target_name, target_is_directory=True)
        else:
            if os.path.isdir(fn):
                shutil.copytree(fn, target_name)
            else:
                shutil.copy(fn, target_name)

    def git_clone(url, target, commit):
        p = subprocess.Popen(['git', 'clone', url, target])
        p.communicate()
        os.chdir(target)
        p = subprocess.Popen(['git', 'checkout', commit])
        p.communicate()

    git_clone('https://github.com/saalfeldlab/CNNectome', os.path.join(path, 'CNNectome'), '0c2220d36423497fdfc68d07beb64e70010e5a75')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--data-pattern', required=True)
    parser.add_argument('--copy-data', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--conda-sh', default='$HOME/miniconda3/etc/profile.d/conda.sh')
    parser.add_argument('--eqip-revision', default=default_revisions['eqip'])
    parser.add_argument('--log-level', default='INFO', choices=('DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'))
    return parser


def create_experiment_main(argv=sys.argv[1:]):
    parser = get_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.getLevelName(args.log_level))
    try:
        create_experiment(
            args.path,
            args.data_pattern,
            create_conda_env=lambda path: create_eqip_environment(
                name=path,
                use_name_as_prefix=True,
                eqip_revision=args.eqip_revision),
            symlink_data=not args.copy_data,
            overwrite=args.overwrite)
    except Exception as e:
        print('Unable to create experiment:', str(e), file=sys.stderr)
        parser.print_help(sys.stderr)


