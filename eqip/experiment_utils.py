#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import re

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
    print(setup_format_string)

    for setup in setups:
        with open(os.path.join(experiment, str(setup), checkpoint)) as f:
            print(setup_format_string.format(setup), '      ', re.findall(r'\d+', f.readline())[0])

if __name__ == "__main__":
    list_latest_checkpoint()
