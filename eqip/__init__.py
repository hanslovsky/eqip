from __future__ import absolute_import, print_function

from .experiment_utils import list_latest_checkpoint, list_latest_snapshot

from .version_info import _version as version



def create_setup():
    import argparse
    import glob
    import os
    import stat
    import sys

    architecture_template=r'''#!/usr/bin/env bash

EXPERIMENT_NAME="$(basename $(realpath $(pwd)/..))"
SETUP_ID="$(basename $(pwd))"
NAME="${EXPERIMENT_NAME}.${SETUP_ID}-mknet"
USER_ID=${UID}
docker rm -f $NAME
#rm snapshots/*

echo "Starting as user ${USER_ID}"
CONTAINER='%(container)s'

nvidia-docker run --rm \
    -u ${USER_ID} \
    -v /groups/turaga:/groups/turaga \
    -v /groups/saalfeld:/groups/saalfeld \
    -v /nrs/saalfeld:/nrs/saalfeld \
    -w ${PWD} \
    --name ${NAME} \
    "${CONTAINER}" \
    /bin/bash -c "export CUDA_VISIBLE_DEVICES=0; %(command)s %(args)s"
'''

    training_template=r'''#!/usr/bin/env bash
WD=$(pwd)
EXPERIMENT_NAME="$(basename $(realpath $(pwd)/..))"
SETUP_ID="$(basename $(pwd))"
NAME="${EXPERIMENT_NAME}.${SETUP_ID}-training"
USER_ID=${UID}
docker rm -f $NAME
#rm snapshots/*
echo "Starting as user ${USER_ID}"
cd /groups/turaga
cd /groups/saalfeld
cd /nrs/saalfeld
cd $WD

CONTAINER='%(container)s'

nvidia-docker run --rm \
    -u ${USER_ID} \
    -v /groups/turaga:/groups/turaga:rshared \
    -v /groups/saalfeld:/groups/saalfeld:rshared \
    -v /nrs/saalfeld:/nrs/saalfeld:rshared \
    -w ${PWD} \
    --name ${NAME} \
    "${CONTAINER}" \
    /bin/bash -c "export CUDA_VISIBLE_DEVICES=$1; %(command)s %(args)s 2>&1 | tee -a logfile"
'''
# python -u $1 --snapshot-every=2000 --mse-iterations=200000 --malis-iterations=400000 --save-checkpoint-every=2000 --pre-cache-size=100 2>&1
    argv = sys.argv[1:]

    indices = [i for i, x in enumerate(argv) if x == '--']
    indices = indices[:2] + [len(argv)] * (2 - len(indices))

    create_setup_args = argv[:indices[0]]
    mknet_args        = argv[indices[0]+1:indices[1]]
    train_args        = argv[indices[1]+1:]

    # eqip_version = version()

    parser = argparse.ArgumentParser()
    parser.add_argument('experiment')
    parser.add_argument('architecture')
    parser.add_argument('training')
    parser.add_argument('--setup', '-s', type=int, default=None)
    parser.add_argument('--mknet-script-name', default='mknet.sh')
    parser.add_argument('--train-script-name', default='train.sh')


    if version.tag() != '':
        parser.add_argument('--container', type=str, required=True)
    else:
        parser.add_argument('--container', type=str, default='hanslovsky/eqip:%s' % version)

    args = parser.parse_args(create_setup_args)

    def find_max_setup_id(dirs):
        dirs = [os.path.basename(os.path.normpath(d)) for d in dirs if os.path.basename(os.path.normpath(d)).isdigit()]
        relevant_dirs = [d for d in dirs if d.isdigit()]
        return -1 if len(relevant_dirs) == 0 else max(int(d) for d in dirs if d.isdigit())

    os.makedirs(args.experiment, exist_ok=True)
    setup_id  = find_max_setup_id(glob.glob(os.path.join(args.experiment, '*'))) + 1 if args.setup is None else args.setup
    setup_id  = str(setup_id)
    setup_dir = os.path.join(args.experiment, setup_id)
    os.mkdir(setup_dir)

    architecture_script = architecture_template % dict(
        container=args.container,
        command=args.architecture,
        args=' '.join(mknet_args))

    training_script = training_template % dict(
        container=args.container,
        command=args.training,
        args=' '.join(train_args))

    with open(os.path.join(args.experiment, setup_id, args.mknet_script_name), 'w') as f:
        f.write(architecture_script)

    with open(os.path.join(args.experiment, setup_id, args.train_script_name), 'w') as f:
        f.write(training_script)

    for fn in glob.glob(os.path.join(args.experiment, setup_id, '*.sh')):
        os.chmod(fn, os.stat(fn).st_mode | stat.S_IEXEC)

    print("Created setup %s for experiment %s" % (setup_id, args.experiment))


