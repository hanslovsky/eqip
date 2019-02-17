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

_architecture_template_no_docker = r'''#!/usr/bin/env bash
if [ -d "${PWD}/conda-env" ]; then
    echo 'activating conda'
    . $HOME/miniconda3/etc/profile.d/conda.sh
    conda activate "${PWD}/conda-env"
    # conda command not exported to subshell
    # https://github.com/conda/conda/issues/7753
fi

echo "Make networks"
CNNECTOME_PATH="$PWD/../CNNectome"
export PYTHONPATH="$CNNECTOME_PATH:$CNNECTOME_PATH/networks:$PYTHONPATH"
%(command)s %(args)s
'''

_training_template_no_docker = r'''#!/usr/bin/env bash
if [ -d "${PWD}/conda-env" ]; then
    echo 'activating conda'
    . $HOME/miniconda3/etc/profile.d/conda.sh
    conda activate "${PWD}/conda-env"
    # conda command not exported to subshell
    # https://github.com/conda/conda/issues/7753
fi

export CUDA_VISIBLE_DEVICES=$1;
echo "Start training with GPU ${CUDA_VISIBLE_DEVICES}"
%(command)s %(args)s 2>&1| tee -a logfile
'''


def make_architecture(container, command, args):
    return _architecture_template % (dict(container=container, command=command, args=args))

def make_training(container, command, args):
    return _training_template % (dict(container=container, command=command, args=args))

def make_architecture_no_docker(command, args):
    return _architecture_template_no_docker % (dict(command=command, args=args))

def make_training_no_docker(command, args):
    return _training_template_no_docker % (dict(command=command, args=args))