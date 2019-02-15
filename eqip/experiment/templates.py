_architecture_template = r'''#!/usr/bin/env bash

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

_training_template = r'''#!/usr/bin/env bash
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


def make_architecture(container, command, args):
    return _architecture_template % (dict(container=container, command=command, args=args))

def make_training(container, command, args):
    return _training_template % (dict(container=container, command=command, args=args))