import argparse
import glob
import math
import os
import stat

_CREATE_SETUP_TEMPLATE = """#!/usr/bin/env python3

import os

from eqip.experiment.affinities_with_glia import _create_setup

here = os.path.abspath(os.path.dirname(__file__))
_create_setup(experiment_dir=here)
"""

def _create_setup(experiment_dir):

    from eqip.experiment import make_architecture, make_training

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
    parser.add_argument('--docker-container', required=True)

    args, unknown = parser.parse_known_args()

    num_affinities = sum(len(n) for n in (
        args.affinity_neighborhood_x,
        args.affinity_neighborhood_y,
        args.affinity_neighborhood_z))

    with open(os.path.join(setup_dir, 'mknet.sh'), 'w') as f:
        f.write(make_architecture(
            container=args.docker_container,
            command='make-affinities-on-interpolated-ground-truth-with-glia',
            args='--num-affinities=%d' % num_affinities))

    ignore_args_parser = argparse.ArgumentParser()
    ignore_args_parser.add_argument('--docker-container', required=False)
    _, train_args = ignore_args_parser.parse_known_args()

    with open(os.path.join(setup_dir, 'train.sh'), 'w') as f:
        f.write(make_training(
            container=args.docker_container,
            command='train-affinities-on-interpolated-ground-truth-with-glia',
            args=' '.join(train_args)))

    os.chmod(os.path.join(setup_dir, 'mknet.sh'), stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP)
    os.chmod(os.path.join(setup_dir, 'train.sh'), stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP)

    print('Created setup %d for experiment %s' % (setup_id, experiment_dir))

def create_experiment(path):
    os.makedirs(path, exist_ok=False)
    with open(os.path.join(path, 'create-setup.py'), 'w') as f:
        f.write(_CREATE_SETUP_TEMPLATE)
        os.chmod(os.path.join(path, 'create-setup.py'), stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP)

if __name__ == "__main__":
    import shutil
    shutil.rmtree('123')
    create_experiment('123')


