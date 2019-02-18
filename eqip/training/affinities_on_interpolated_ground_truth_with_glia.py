from __future__ import print_function, division

from .. import gunpowder_utils
from .. import io_keys
from .. import tf_util

import logging

from fuse.map_numpy_array import MapNumpyArray

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

from fuse import ElasticAugment, Misalign, SimpleAugment, Snapshot, DefectAugment
from gunpowder import Hdf5Source, Coordinate, BatchRequest, Normalize, Pad, RandomLocation, Reject, \
    ArraySpec, IntensityAugment, RandomProvider, GrowBoundary, IntensityScaleShift, \
    PrintProfilingStats, build, PreCache
from gunpowder.contrib import \
    ZeroOutConstSections
from gunpowder.nodes import \
    AddAffinities,\
    RenumberConnectedComponents,\
    BalanceLabels
from gunpowder.tensorflow import Train
import tensorflow as tf
import os
import math
import numpy as np
import json
import sys

RAW_KEY              = gunpowder_utils.RAW_KEY
DEFECT_MASK_KEY      = gunpowder_utils.DEFECT_MASK_KEY
LABELS_KEY           = gunpowder_utils.NEURON_IDS_NO_GLIA_KEY
GT_MASK_KEY          = gunpowder_utils.MASK_KEY
LOSS_GRADIENT_KEY    = gunpowder_utils.LOSS_GRADIENT_KEY
AFFINITIES_KEY       = gunpowder_utils.AFFINITIES_KEY
GT_AFFINITIES_KEY    = gunpowder_utils.GT_AFFINITIES_KEY
AFFINITIES_MASK_KEY  = gunpowder_utils.AFFINITIES_MASK_KEY
AFFINITIES_SCALE_KEY = gunpowder_utils.AFFINITIES_SCALE_KEY
GLIA_MASK_KEY        = gunpowder_utils.GLIA_MASK_KEY
GLIA_KEY             = gunpowder_utils.GLIA_KEY
GLIA_SCALE_KEY       = gunpowder_utils.GLIA_SCALE_KEY
GT_GLIA_KEY          = gunpowder_utils.GT_GLIA_KEY

NETWORK_INPUT_SHAPE  = Coordinate((43, 430, 430))
NETWORK_OUTPUT_SHAPE = Coordinate((65,  70,  70))

INPUT_VOXEL_SIZE  = Coordinate((360.0,  36.0,  36.0))
OUTPUT_VOXEL_SIZE = Coordinate((120.0, 108.0, 108.0))

NETWORK_INPUT_SHAPE_WORLD  = INPUT_VOXEL_SIZE  * NETWORK_INPUT_SHAPE
NETWORK_OUTPUT_SHAPE_WORLD = OUTPUT_VOXEL_SIZE * NETWORK_OUTPUT_SHAPE

SHAPE_DIFF_WORLD  = NETWORK_INPUT_SHAPE_WORLD - NETWORK_OUTPUT_SHAPE_WORLD
SHAPE_DIFF_INPUT  = SHAPE_DIFF_WORLD / INPUT_VOXEL_SIZE
SHAPE_DIFF_OUTPUT = SHAPE_DIFF_WORLD / OUTPUT_VOXEL_SIZE
PADDING_OUTPUT    = SHAPE_DIFF_OUTPUT / 2

# TODO fix padding

def train_until(
        data_providers,
        affinity_neighborhood,
        meta_graph_filename,
        stop,
        input_shape,
        output_shape,
        loss,
        optimizer,
        tensor_affinities,
        tensor_affinities_mask,
        tensor_glia,
        tensor_glia_mask,
        summary,
        save_checkpoint_every,
        pre_cache_size,
        pre_cache_num_workers,
        snapshot_every,
        balance_labels,
        renumber_connected_components,
        network_inputs,
        ignore_labels_for_slip,
        grow_boundaries,
        snapshot_dir):

    ignore_keys_for_slip = (LABELS_KEY, GT_MASK_KEY, GT_GLIA_KEY, GLIA_MASK_KEY) if ignore_labels_for_slip else ()

    defect_dir = '/groups/saalfeld/home/hanslovskyp/experiments/quasi-isotropic/data/defects'
    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
        print('Resuming training from', trained_until)
    else:
        trained_until = 0
        print('Starting fresh training')

    input_voxel_size = Coordinate((120, 12, 12)) * 3
    output_voxel_size = Coordinate((40, 36, 36)) * 3

    input_size = Coordinate(input_shape) * input_voxel_size
    output_size = Coordinate(output_shape) * output_voxel_size

    num_affinities = sum(len(nh) for nh in affinity_neighborhood)
    gt_affinities_size = Coordinate((num_affinities,) + tuple(s for s in output_size))
    print("gt affinities size", gt_affinities_size)

    # TODO why is GT_AFFINITIES three-dimensional? compare to
    # TODO https://github.com/funkey/gunpowder/blob/master/examples/cremi/train.py#L35
    # TODO Use glia scale somehow, probably not possible with tensorflow 1.3 because it does not know uint64...
    # specifiy which Arrays should be requested for each batch
    request = BatchRequest()
    request.add(RAW_KEY,             input_size,  voxel_size=input_voxel_size)
    request.add(LABELS_KEY,          output_size, voxel_size=output_voxel_size)
    request.add(GT_AFFINITIES_KEY,   output_size, voxel_size=output_voxel_size)
    request.add(AFFINITIES_MASK_KEY, output_size, voxel_size=output_voxel_size)
    request.add(GT_MASK_KEY,         output_size, voxel_size=output_voxel_size)
    request.add(GLIA_MASK_KEY,       output_size, voxel_size=output_voxel_size)
    request.add(GLIA_KEY,            output_size, voxel_size=output_voxel_size)
    request.add(GT_GLIA_KEY,         output_size, voxel_size=output_voxel_size)
    if balance_labels:
        request.add(AFFINITIES_SCALE_KEY, output_size, voxel_size=output_voxel_size)
    # always balance glia labels!
    request.add(GLIA_SCALE_KEY, output_size, voxel_size=output_voxel_size)
    network_inputs[tensor_affinities_mask] = AFFINITIES_SCALE_KEY if balance_labels else AFFINITIES_MASK_KEY
    network_inputs[tensor_glia_mask]       = GLIA_SCALE_KEY#GLIA_SCALE_KEY if balance_labels else GLIA_MASK_KEY

    # create a tuple of data sources, one for each HDF file
    data_sources = tuple(
        provider +
        Normalize(RAW_KEY) + # ensures RAW is in float in [0, 1]

        # zero-pad provided RAW and GT_MASK to be able to draw batches close to
        # the boundary of the available data
        # size more or less irrelevant as followed by Reject Node
        Pad(RAW_KEY, None) +
        Pad(GT_MASK_KEY, None) +
        Pad(GLIA_MASK_KEY, None) +
        Pad(LABELS_KEY, size=NETWORK_OUTPUT_SHAPE / 2, value=np.uint64(-3)) +
        Pad(GT_GLIA_KEY, size=NETWORK_OUTPUT_SHAPE / 2) +
        # Pad(LABELS_KEY, None) +
        # Pad(GT_GLIA_KEY, None) +
        RandomLocation() + # chose a random location inside the provided arrays
        Reject(mask=GT_MASK_KEY, min_masked=0.5) +
        Reject(mask=GLIA_MASK_KEY, min_masked=0.5) +
        MapNumpyArray(lambda array: np.require(array, dtype=np.int64), GT_GLIA_KEY)
        # NumpyRequire(GT_GLIA_KEY, dtype=np.int64) # this is necessary because gunpowder 1.3 only understands int64, not uint64

        for provider in data_providers)

    # TODO figure out what this is for
    snapshot_request = BatchRequest({
        LOSS_GRADIENT_KEY : request[GT_AFFINITIES_KEY],
        AFFINITIES_KEY    : request[GT_AFFINITIES_KEY],
    })

    # no need to do anything here. random sections will be replaced with sections from this source (only raw)
    artifact_source = (
        Hdf5Source(
            os.path.join(defect_dir, 'sample_ABC_padded_20160501.defects.hdf'),
            datasets={
                RAW_KEY        : 'defect_sections/raw',
                DEFECT_MASK_KEY : 'defect_sections/mask',
            },
            array_specs={
                RAW_KEY        : ArraySpec(voxel_size=input_voxel_size),
                DEFECT_MASK_KEY : ArraySpec(voxel_size=input_voxel_size),
            }
        ) +
        RandomLocation(min_masked=0.05, mask=DEFECT_MASK_KEY) +
        Normalize(RAW_KEY) +
        IntensityAugment(RAW_KEY, 0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        ElasticAugment(
            voxel_size=(360, 36, 36),
            control_point_spacing=(4, 40, 40),
            control_point_displacement_sigma=(0, 2 * 36, 2 * 36),
            rotation_interval=(0, math.pi / 2.0),
            subsample=8
        ) +
        SimpleAugment(transpose_only=[1,2])
    )

    train_pipeline  = data_sources
    train_pipeline += RandomProvider()

    train_pipeline += ElasticAugment(
            voxel_size=(360, 36, 36),
            control_point_spacing=(4, 40, 40),
            control_point_displacement_sigma=(0, 2 * 36, 2 * 36),
            rotation_interval=(0, math.pi / 2.0),
            augmentation_probability=0.5,
            subsample=8
        )

    # train_pipeline += Log.log_numpy_array_stats_after_process(GT_MASK_KEY, 'min', 'max', 'dtype', logging_prefix='%s: before misalign: ' % GT_MASK_KEY)
    train_pipeline += Misalign(z_resolution=360, prob_slip=0.05, prob_shift=0.05, max_misalign=(360,) * 2, ignore_keys_for_slip=ignore_keys_for_slip)
    # train_pipeline += Log.log_numpy_array_stats_after_process(GT_MASK_KEY, 'min', 'max', 'dtype', logging_prefix='%s: after  misalign: ' % GT_MASK_KEY)

    train_pipeline += SimpleAugment(transpose_only=[1,2])
    train_pipeline += IntensityAugment(RAW_KEY, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)
    train_pipeline += DefectAugment(RAW_KEY,
                                    prob_missing=0.03,
                                    prob_low_contrast=0.01,
                                    prob_artifact=0.03,
                                    artifact_source=artifact_source,
                                    artifacts=RAW_KEY,
                                    artifacts_mask=DEFECT_MASK_KEY,
                                    contrast_scale=0.5)
    train_pipeline += IntensityScaleShift(RAW_KEY, 2, -1)
    train_pipeline += ZeroOutConstSections(RAW_KEY)

    if grow_boundaries > 0:
        train_pipeline += GrowBoundary(LABELS_KEY, GT_MASK_KEY, steps=grow_boundaries, only_xy=True)

    _logger.info("Renumbering connected components? %s", renumber_connected_components)
    if renumber_connected_components:
        train_pipeline += RenumberConnectedComponents(labels=LABELS_KEY)


    train_pipeline += AddAffinities(
            affinity_neighborhood=affinity_neighborhood,
            labels=LABELS_KEY,
            labels_mask=GT_MASK_KEY,
            affinities=GT_AFFINITIES_KEY,
            affinities_mask=AFFINITIES_MASK_KEY)

    snapshot_datasets = {
        RAW_KEY: 'volumes/raw',
        LABELS_KEY: 'volumes/labels/neuron_ids',
        GT_AFFINITIES_KEY: 'volumes/affinities/gt',
        GT_GLIA_KEY: 'volumes/labels/glia_gt',
        AFFINITIES_KEY: 'volumes/affinities/prediction',
        LOSS_GRADIENT_KEY: 'volumes/loss_gradient',
        AFFINITIES_MASK_KEY: 'masks/affinities',
        GLIA_KEY: 'volumes/labels/glia_pred',
        GT_MASK_KEY: 'masks/gt',
        GLIA_MASK_KEY: 'masks/glia'}

    if balance_labels:
        train_pipeline += BalanceLabels(labels=GT_AFFINITIES_KEY, scales=AFFINITIES_SCALE_KEY, mask=AFFINITIES_MASK_KEY)
        snapshot_datasets[AFFINITIES_SCALE_KEY] = 'masks/affinitiy-scale'
    train_pipeline += BalanceLabels(labels=GT_GLIA_KEY, scales=GLIA_SCALE_KEY, mask=GLIA_MASK_KEY)
    snapshot_datasets[GLIA_SCALE_KEY] = 'masks/glia-scale'


    if (pre_cache_size > 0 and pre_cache_num_workers > 0):
        train_pipeline += PreCache(cache_size=pre_cache_size, num_workers=pre_cache_num_workers)
    train_pipeline += Train(
            summary=summary,
            graph=meta_graph_filename,
            save_every=save_checkpoint_every,
            optimizer=optimizer,
            loss=loss,
            inputs=network_inputs,
            log_dir='log',
            outputs={tensor_affinities: AFFINITIES_KEY, tensor_glia: GLIA_KEY},
            gradients={tensor_affinities: LOSS_GRADIENT_KEY},
            array_specs={
                AFFINITIES_KEY       : ArraySpec(voxel_size=output_voxel_size),
                LOSS_GRADIENT_KEY    : ArraySpec(voxel_size=output_voxel_size),
                AFFINITIES_MASK_KEY  : ArraySpec(voxel_size=output_voxel_size),
                GT_MASK_KEY          : ArraySpec(voxel_size=output_voxel_size),
                AFFINITIES_SCALE_KEY : ArraySpec(voxel_size=output_voxel_size),
                GLIA_MASK_KEY        : ArraySpec(voxel_size=output_voxel_size),
                GLIA_SCALE_KEY       : ArraySpec(voxel_size=output_voxel_size),
                GLIA_KEY             : ArraySpec(voxel_size=output_voxel_size)
            }
        )

    train_pipeline += Snapshot(
            snapshot_datasets,
            every=snapshot_every,
            output_filename='batch_{iteration}.hdf',
            output_dir=snapshot_dir,
            additional_request=snapshot_request,
            attributes_callback=Snapshot.default_attributes_callback())

    train_pipeline += PrintProfilingStats(every=50)

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(trained_until, stop):
            b.request_batch(request)

    print("Training finished")


def train(argv=sys.argv[1:]):

    import argparse

    def bounded_integer(val, lower=None, upper=None):
        val = int(val)
        if lower is not None and val < lower or upper is not None and val > upper:
            raise argparse.ArgumentTypeError('Value %d is out of bounds for [%s, %s]' % (val, str(-math.inf if lower is None else lower), str(math.inf if upper is None else upper)))
        return val

    def make_default_data_provider_string():
        data_dir = os.path.join(os.pardir, 'data')
        file_pattern = '*'
        return '{}:{}={}:{}={}:{}={}:{}={}:{}={}'.format(
            os.path.join(data_dir, file_pattern),
            'RAW',               gunpowder_utils.DEFAULT_PATHS['raw'],
            'NEURON_IDS_NOGLIA', gunpowder_utils.DEFAULT_PATHS['neuron_ids_noglia'],
            'MASK',              'volumes/labels/mask-downsampled',
            'GLIA_MASK',         'volumes/labels/mask-downsampled',
            'GT_GLIA',           gunpowder_utils.DEFAULT_PATHS['gt_glia'])

    default_data_provider_string = make_default_data_provider_string()

    parser = argparse.ArgumentParser()
    parser.add_argument('--training-directory', default='.', type=str)
    parser.add_argument('--snapshot-every', default=500, type=lambda arg: bounded_integer(arg, 1))
    parser.add_argument('--meta-graph-filename', default='unet', type=str, help='Filename with information about meta graph for network.')
    parser.add_argument('--mse-iterations', type=lambda arg: bounded_integer(arg, 0), default=100000, help='Number of iterations to pre-train with mean square error loss', metavar='MSE_ITERATIONS')
    parser.add_argument('--malis-iterations', type=lambda arg: bounded_integer(arg, 0), default=400000, help='Number of iterations to train with malis loss', metavar='MALIS_ITERATIONS')
    parser.add_argument('--input-shape', type=int, nargs=3, default=(43, 430, 430))
    parser.add_argument('--output-shape', type=int, nargs=3, default=(65, 70, 70))
    parser.add_argument('--log-level', choices=('DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'), default='INFO', type=str)
    parser.add_argument('--net-io-names', type=str, default='net_io_names.json', help='Path to file holding network input/output name specs')
    parser.add_argument('--save-checkpoint-every', type=lambda arg: bounded_integer(arg, 1), default=2000, metavar='N_BETWEEN_CHECKPOINTS', help='Make a checkpoint of the model every Nth iteration.')
    parser.add_argument('--pre-cache-num-workers', type=lambda arg: bounded_integer(arg, 1), default=1, metavar='PRECACHE_NUM_WORKERS', help='Number of workers used to populate pre-cache')
    parser.add_argument('--pre-cache-size', type=lambda arg: bounded_integer(arg, 0), default=1, metavar='PRECACHE_SIZE', help='Size of pre-cache')
    parser.add_argument('--ignore-labels-for-slip', action='store_true')
    parser.add_argument('--affinity-neighborhood-x', nargs='+', type=int, default=(-1,))
    parser.add_argument('--affinity-neighborhood-y', nargs='+', type=int, default=(-1,))
    parser.add_argument('--affinity-neighborhood-z', nargs='+', type=int, default=(-1,))
    parser.add_argument('--grow-boundaries', metavar='STEPS', type=int, default=0, help='Grow boundaries for STEPS if larger than 0.')
    parser.add_argument('--data-provider', metavar='GLOB_PATTERN[:raw=<dataset>[:neuron_ids_noglia=<dataset>[:mask=<dataset>[:glia_gt=<dataset>]]]]', default=(default_data_provider_string,), nargs='+', help='Add data provider.')

    args = parser.parse_args(argv)
    log_levels=dict(DEBUG=logging.DEBUG, INFO=logging.INFO, WARN=logging.WARN, ERROR=logging.ERROR, CRITICAL=logging.CRITICAL)

    if args.mse_iterations % args.save_checkpoint_every != 0:
        parser.error('MSE_ITERATIONS %d is not integer multiple of N_BETWEEN_CHECKPOINTS %d' % (args.mse_iterations, args.save_checkpoint_every))
    if args.malis_iterations % args.save_checkpoint_every != 0:
        parser.error('MALIS_ITERATIONS %d is not integer multiple of N_BETWEEN_CHECKPOINTS %d' % (args.malis_iterations, args.save_checkpoint_every))

    with open(args.net_io_names, 'r') as f:
        net_io_names = json.load(f)

    save_checkpoint_every = args.save_checkpoint_every

    neighborhood = [] \
        + [[nh, 0, 0] for nh in args.affinity_neighborhood_z] \
        + [[0, nh, 0] for nh in args.affinity_neighborhood_y] \
        + [[0, 0, nh] for nh in args.affinity_neighborhood_x]

    print('Using neighborhood', neighborhood)

    logging.basicConfig(level=log_levels[args.log_level])

    _logger.info('Got data providers from user: %s', args.data_provider)
    data_providers = gunpowder_utils.make_data_providers(*args.data_provider, required_paths=('raw', 'neuron_ids_noglia', 'mask', 'glia_mask', 'gt_glia'))
    _logger.info("Data providers: %s'", data_providers)


    if tf.train.latest_checkpoint('.'):
        trained_until = int(tf.train.latest_checkpoint('.').split('_')[-1])
        print('Resuming training from', trained_until)
    else:
        trained_until = 0
        print('Starting fresh training')

    inputs = {
        net_io_names[io_keys.RAW]           : RAW_KEY,
        net_io_names[io_keys.GT_AFFINITIES] : GT_AFFINITIES_KEY,
        net_io_names[io_keys.GT_LABELS]     : LABELS_KEY,
        net_io_names[io_keys.GT_GLIA]       : GT_GLIA_KEY,
    }

    snapshot_dir = os.path.join(args.training_directory, 'snapshots')

    if trained_until < args.mse_iterations:

        mse_loss = tf_util.loss_affinities_with_glia(
            net_io_names=net_io_names,
            optimizer_or_name='glia-mse-affinities-mse-loss-optimizer',
            summary_name='glia-mse-affinities-mse-loss')

        train_until(
            data_providers=data_providers,
            affinity_neighborhood=neighborhood,
            meta_graph_filename=args.meta_graph_filename,
            stop=args.mse_iterations - trained_until,
            input_shape=args.input_shape,
            output_shape=args.output_shape,
            loss=None,
            optimizer=mse_loss,
            summary='glia-mse-affinities-mse-loss:0',
            tensor_affinities=net_io_names[io_keys.AFFINITIES],
            tensor_affinities_mask=net_io_names[io_keys.AFFINITIES_MASK],
            tensor_glia=net_io_names[io_keys.GLIA],
            tensor_glia_mask=net_io_names[io_keys.GLIA_MASK],
            save_checkpoint_every=args.save_checkpoint_every,
            pre_cache_size=args.pre_cache_size,
            pre_cache_num_workers=args.pre_cache_num_workers,
            snapshot_every=args.snapshot_every,
            balance_labels=True,
            renumber_connected_components=False,
            network_inputs=inputs,
            ignore_labels_for_slip=args.ignore_labels_for_slip,
            grow_boundaries=args.grow_boundaries,
            snapshot_dir=snapshot_dir)

    if tf.train.latest_checkpoint('.') and int(tf.train.latest_checkpoint('.').split('_')[-1]) < args.mse_iterations:
        raise Exception("Inconsistency!")

    malis_loss =tf_util.malis_loss_with_glia(
        net_io_names=net_io_names,
        neighborhood=neighborhood,
        optimizer_or_name='glia-mse-affinities-mse-loss-optimizer',
        summary_name='glia-mse-affinities-malis-loss')

    train_until(
            data_providers=data_providers,
        affinity_neighborhood=neighborhood,
        meta_graph_filename=args.meta_graph_filename,
        stop=args.malis_iterations - (trained_until - args.mse_iterations),
        input_shape=args.input_shape,
        output_shape=args.output_shape,
        loss=None,
        optimizer=malis_loss,
        summary='glia-mse-affinities-malis-loss:0',
        tensor_affinities=net_io_names[io_keys.AFFINITIES],
        tensor_affinities_mask=net_io_names[io_keys.AFFINITIES_MASK],
        tensor_glia=net_io_names[io_keys.GLIA],
        tensor_glia_mask=net_io_names[io_keys.GLIA_MASK],
        save_checkpoint_every=args.save_checkpoint_every,
        pre_cache_size=args.pre_cache_size,
        pre_cache_num_workers=args.pre_cache_num_workers,
        snapshot_every=args.snapshot_every,
        balance_labels=False,
        renumber_connected_components=True,
        network_inputs=inputs,
        ignore_labels_for_slip=args.ignore_labels_for_slip,
        grow_boundaries=args.grow_boundaries,
        snapshot_dir=snapshot_dir)
