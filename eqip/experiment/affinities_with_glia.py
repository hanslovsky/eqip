import argparse
import glob
import logging
import math
import os
import shutil
import stat
import sys

from ..conda import clone_eqip_environment, create_eqip_environment, default_revisions

_CREATE_SETUP_TEMPLATE = """#!/usr/bin/env python3

import os

from eqip.experiment.affinities_with_glia import _create_setup

here = os.path.abspath(os.path.dirname(__file__))
_create_setup(experiment_dir=here)
"""

_PREDICTION_SCRIPT=r'''#!/usr/bin/env bash
set -e

THIS_DIR=$(dirname $0)

function _exit_with_error() {
    echo "$1" >&2
    exit "${2:-1}"
}

function _exit_no_argparse() {
    _exit_with_error \
        "Get argparse.bash from https://github.com/nhoffman/argparse-bash" \
        1
}

function _get_and_source_argparse() {
    echo "Downloading argparse.bash..."
    O_FILE=${1:-argparse.bash}
    wget https://raw.githubusercontent.com/nhoffman/argparse-bash/master/argparse.bash -O $O_FILE 2>/dev/null
    chmod +x $O_FILE
    source $O_FILE
}

ARGPARSE_DESCRIPTION="Predict. See also predict-daisy --help"
ARGPARSE_BASH=$THIS_DIR/argparse.bash
NAME=$(head -n1 $THIS_DIR/name 2>/dev/null || echo '')
source $ARGPARSE_BASH 2>/dev/null || _get_and_source_argparse $ARGPARSE_BASH || _exit_no_argparse
argparse "$@" <<EOF || exit 1

parser.add_argument('--input-container', required=True)
parser.add_argument('--input', required=True)
parser.add_argument('--output-container', required=True)
parser.add_argument('--gpus', required=True, help='comma-separated list of gpus')
parser.add_argument('--num-channels', required=True, type=int)
parser.add_argument('--num-workers', required=False, type=int, default=1)
parser.add_argument('--prediction-prefix', required=False, help='Defaults to volumes/predictions/<NAME>/<SETUP>/<ITERATION>')
parser.add_argument('--setup', required=True, type=int)
parser.add_argument('--iteration', required=False, type=int, help='Will default to latest checkpoint if not specified')
parser.add_argument('--name', required=False, help='Defaults to contents of ${THIS_DIR}/name', default='$NAME')
parser.add_argument('--conda-sh', required=False, default='$HOME/miniconda3/etc/profile.d/conda.sh')
parser.add_argument('--conda-environment', required=False, default='${THIS_DIR}/conda-env')
EOF

[ -n "$NAME" ] || _exit_with_error "No name specified or found in ${THIS_DIR}/name" 1


source $(realpath $CONDA_SH)
conda activate $(realpath $CONDA_ENVIRONMENT)

ITERATION=${ITERATION:-$(head -n1 ${THIS_DIR}/${SETUP}/checkpoint | sed -r 's/^.*unet_checkpoint_([0-9]+).*/\1/')}
PREDICTION_PREFIX=${PREDICTION_PREFIX:-volumes/predictions/$NAME/$SETUP/$ITERATION}

predict-daisy \
    --input-container=$INPUT_CONTAINER \
    --input $INPUT Placeholder:0 \
    --output-container=$OUTPUT_CONTAINER \
    --gpus $(echo $GPUS | tr ',' ' ') \
    --output ${PREDICTION_PREFIX}/affinities float32 ${NUM_CHANNELS} Slice:0 \
    --output ${PREDICTION_PREFIX}/glia       float32 0               Slice_1:0 \
    --experiment-directory=${THIS_DIR}/${SETUP} \
    --iteration=$ITERATION \
    --num-workers=${NUM_WORKERS}
'''

_AVERAGE_AFFINITIES_SCRIPT=r'''#!/usr/bin/env bash
set -e

THIS_DIR=$(dirname $0)

function _exit_with_error() {
    echo "$1" >&2
    exit "${2:-1}"
}

function _exit_no_argparse() {
    _exit_with_error \
        "Get argparse.bash from https://github.com/nhoffman/argparse-bash" \
        1
}

function _get_and_source_argparse() {
    echo "Downloading argparse.bash..."
    O_FILE=${1:-argparse.bash}
    wget https://raw.githubusercontent.com/nhoffman/argparse-bash/master/argparse.bash -O $O_FILE 2>/dev/null
    chmod +x $O_FILE
    source $O_FILE
}

ARGPARSE_DESCRIPTION="Predict. See also predict-daisy --help"
ARGPARSE_BASH=$THIS_DIR/argparse.bash
NAME=$(head -n1 $THIS_DIR/name 2>/dev/null || echo '')
source $ARGPARSE_BASH 2>/dev/null || _get_and_source_argparse $ARGPARSE_BASH || _exit_no_argparse
argparse "$@" <<EOF || exit 1

parser.add_argument('--container', required=True)
parser.add_argument('--mask-container', required=True)
parser.add_argument('--mask', required=True)
parser.add_argument('--n-nodes', required=True, help='Number of spark nodes', type=int)
parser.add_argument('--prefix', required=False, help='Defaults to volumes/predictions/<NAME>/<SETUP>/<ITERATION>')
parser.add_argument('--setup', required=True, type=int)
parser.add_argument('--iteration', required=False, type=int, help='Will default to latest checkpoint if not specified')
parser.add_argument('--name', required=False, help='Defaults to contents of ${THIS_DIR}/name', default='$NAME')
parser.add_argument('--glia-mask-threshold', required=False, type=float)
parser.add_argument('--use-glia-mask', required=False, action='store_true')
parser.add_argument('--jar', required=False, default='${HOME}/label-utilities-spark-0.7.1-shaded.jar')
parser.add_argument('--flintstone', required=False, default='$HOME/flintstone/flintstone.sh')
parser.add_argument('--block-size', required=False, default='64,64,64')
parser.add_argument('--blocks-per-task', required=False, default='4,4,4')
EOF

[ -n "$NAME" ] || _exit_with_error "No name specified or found in ${THIS_DIR}/name" 1

CLASS="org.janelia.saalfeldlab.label.spark.affinities.AverageAffinities"


ITERATION=${ITERATION:-$(head -n1 ${THIS_DIR}/${SETUP}/checkpoint | sed -r 's/^.*unet_checkpoint_([0-9]+).*/\1/')}
PREFIX=${PREFIX:-volumes/predictions/$NAME/$SETUP/$ITERATION}

[ -f $FLINSTONE ] || _exit_with_error "Flintstone not found at $FLINTSTONE" 1
[ -f "${THIS_DIR}/${SETUP}/offsets" ] || _exit_with_error "No offsets specified in ${THIS_DIR}/${SETUP}/offsets" 1

CMD="$FLINTSTONE \
    ${N_NODES} \
    ${JAR} \
    ${CLASS} \
    $CONTAINER \
    --affinity-dataset=$PREFIX/affinities \
    --mask-container=$MASK_CONTAINER \
    --mask-dataset=$MASK \
    --block-size=$BLOCK_SIZE \
    --blocks-per-task=$BLOCKS_PER_TASK"

for OFFSET in $(cat "${THIS_DIR}/${SETUP}/offsets"); do
    # echo $OFFSET
    CMD="$CMD --offsets=$OFFSET"
done


if [ -n "$USE_GLIA_MASK" ]; then
    CMD="$CMD --glia-mask-dataset=$PREFIX/glia"
    if [ -n "$GLIA_MASK_THRESHOLD" ]; then
        CMD="$CMD --glia-mask-threshold=$GLIA_MASK_THRESHOLD"
    fi
fi

$CMD

'''

_WATERSHED_SCRIPT=r'''#!/usr/bin/env bash
set -e

THIS_DIR=$(dirname $0)

function _exit_with_error() {
    echo "$1" >&2
    exit "${2:-1}"
}

function _exit_no_argparse() {
    _exit_with_error \
        "Get argparse.bash from https://github.com/nhoffman/argparse-bash" \
        1
}

function _get_and_source_argparse() {
    echo "Downloading argparse.bash..."
    O_FILE=${1:-argparse.bash}
    wget https://raw.githubusercontent.com/nhoffman/argparse-bash/master/argparse.bash -O $O_FILE 2>/dev/null
    chmod +x $O_FILE
    source $O_FILE
}

ARGPARSE_DESCRIPTION="Predict. See also predict-daisy --help"
ARGPARSE_BASH=$THIS_DIR/argparse.bash
NAME=$(head -n1 $THIS_DIR/name 2>/dev/null || echo '')
source $ARGPARSE_BASH 2>/dev/null || _get_and_source_argparse $ARGPARSE_BASH || _exit_no_argparse
argparse "$@" <<EOF || exit 1

parser.add_argument('--container', required=True)
parser.add_argument('--n-nodes', required=True, help='Number of spark nodes', type=int)
parser.add_argument('--prefix', required=False, help='Defaults to volumes/predictions/<NAME>/<SETUP>/<ITERATION>')
parser.add_argument('--setup', required=True, type=int)
parser.add_argument('--iteration', required=False, type=int, help='Will default to latest checkpoint if not specified')
parser.add_argument('--name', required=False, help='Defaults to contents of ${THIS_DIR}/name', default='$NAME')
parser.add_argument('--jar', required=False, default='${HOME}/label-utilities-spark-0.7.1-shaded.jar')
parser.add_argument('--flintstone', required=False, default='$HOME/flintstone/flintstone.sh')
parser.add_argument('--block-size', required=False, default='64,64,64')
parser.add_argument('--blocks-per-task', required=False, default='4,4,4')
parser.add_argument('--threshold', required=True, type=float)
parser.add_argument('--minimum-watershed-affinity', required=True, type=float)
EOF

[ -n "$NAME" ] || _exit_with_error "No name specified or found in ${THIS_DIR}/name" 1

CLASS="org.janelia.saalfeldlab.label.spark.watersheds.SparkWatersheds"


ITERATION=${ITERATION:-$(head -n1 ${THIS_DIR}/${SETUP}/checkpoint | sed -r 's/^.*unet_checkpoint_([0-9]+).*/\1/')}
PREFIX=${PREFIX:-volumes/predictions/$NAME/$SETUP/$ITERATION}

[ -f $FLINSTONE ] || _exit_with_error "Flintstone not found at $FLINTSTONE" 1
[ -f "${THIS_DIR}/${SETUP}/offsets" ] || _exit_with_error "No offsets specified in ${THIS_DIR}/${SETUP}/offsets" 1

$FLINTSTONE \
    ${N_NODES} \
    ${JAR} \
    ${CLASS} \
    $CONTAINER \
    --averaged-affinity-dataset=$PREFIX/affinities-averaged \
    --label-datasets-prefix=$PREFIX/watersheds/merge_threshold=${THRESHOLD}_seed_threshold=${MINIMUM_WATERSHED_AFFINITY} \
    --halo=0,0,0 \
    --block-size=$BLOCK_SIZE \
    --blocks-per-task=$BLOCKS_PER_TASK \
    --relabel=true \
    --threshold=$THRESHOLD \
    --minimum-watershed-affinity=$MINIMUM_WATERSHED_AFFINITY


'''

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
    parser.add_argument('--data-provider', required=False, default=os.path.join(experiment_dir, 'data/*:MASK=volumes/labels/mask-downsampled-75%-y:GLIA_MASK=volumes/labels/mask-downsampled-75%-y'))
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

    def as_vector(offset, dimension):
        return tuple(offset if d == dimension else 0 for d in range(3))

    offsets = tuple(as_vector(o, 2) for o in args.affinity_neighborhood_z) \
            + tuple(as_vector(o, 1) for o in args.affinity_neighborhood_y) \
            + tuple(as_vector(o, 0) for o in args.affinity_neighborhood_y)

    with open(os.path.join(setup_dir, 'offsets'), 'w') as f:
        f.write('\n'.join(('%d,%d,%d' % o) + (':%d' %i) for i, o in enumerate(offsets)))
    os.chmod(os.path.join(setup_dir, 'offsets'), stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

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
        name=None,
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

    name = os.path.basename(os.path.normpath(path)) if name is None else name

    data_dir      = os.path.join(path, 'data')
    conda_env_dir = os.path.join(path, 'conda-env')
    os.makedirs(data_dir, exist_ok=False)
    create_conda_env(conda_env_dir)
    with open(os.path.join(path, 'create-setup.py'), 'w') as f:
        f.write(_CREATE_SETUP_TEMPLATE)
    os.chmod(os.path.join(path, 'create-setup.py'), stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP)

    with open(os.path.join(path, 'data-source'), 'w') as f:
        f.write(data_pattern)

    with open(os.path.join(path, 'name'), 'w') as f:
        f.write(name)

    with open(os.path.join(path, 'predict.sh'), 'w') as f:
        f.write(_PREDICTION_SCRIPT)
    os.chmod(os.path.join(path, 'predict.sh'), stat.S_IXUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    with open(os.path.join(path, 'average-affinities.sh'), 'w') as f:
        f.write(_AVERAGE_AFFINITIES_SCRIPT)
    os.chmod(os.path.join(path, 'average-affinities.sh'), stat.S_IXUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    with open(os.path.join(path, 'watersheds.sh'), 'w') as f:
        f.write(_WATERSHED_SCRIPT)
    os.chmod(os.path.join(path, 'watersheds.sh'), stat.S_IXUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

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

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--experiment-name', required=False, help='Defaults to basename of PATH')
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
            overwrite=args.overwrite,
            name=args.experiment_name)
    except Exception as e:
        print('Unable to create experiment:', str(e), file=sys.stderr)
        parser.print_help(sys.stderr)


