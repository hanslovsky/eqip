#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import re


def list_latest_snapshot():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', '-e', default='.')
    parser.add_argument('--setup', nargs='+', required=False, type=int)
    parser.add_argument('--snapshots-directory', required=False, default='snapshots')

    args                = parser.parse_args()
    experiment          = args.experiment
    snapshots_directory = args.snapshots_directory

    if args.setup is None or len(args.setup) == 0:
        setups = sorted(tuple(int(d) for d in os.listdir(experiment) if d.isdigit() and os.path.isdir(os.path.join(experiment, d, snapshots_directory))))
    else:
        setups = args.setup

    max_digits = max(len(str(x)) for x in setups)
    setup_format_string = '{:%dd}' % max_digits

    for setup in setups:
        files  = tuple(re.findall(r'\d+', f) for f in os.listdir(os.path.join(experiment, str(setup), snapshots_directory)) if f.endswith('.hdf'))
        latest = max(tuple(int(f[0]) for f in files if len(f) > 0))
        print(setup_format_string.format(setup), '      ', latest)


def list_latest_checkpoint():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', '-e', default='.')
    parser.add_argument('--setup', nargs='+', required=False, type=int)
    parser.add_argument('--checkpoint-filename', required=False, default='checkpoint')

    args       = parser.parse_args()
    experiment = args.experiment
    checkpoint = args.checkpoint_filename

    if args.setup is None or len(args.setup) == 0:
        setups = sorted(tuple(int(d) for d in os.listdir(experiment) if d.isdigit() and os.path.isdir(os.path.join(experiment, d)) and os.path.exists(os.path.join(experiment, d, checkpoint))))
    else:
        setups = args.setup

    max_digits = max(len(str(x)) for x in setups)
    setup_format_string = '{:%dd}' % max_digits

    for setup in setups:
        with open(os.path.join(experiment, str(setup), checkpoint)) as f:
            print(setup_format_string.format(setup), '      ', re.findall(r'\d+', f.readline())[0])

if __name__ == "__main__":
    list_latest_snapshot()
