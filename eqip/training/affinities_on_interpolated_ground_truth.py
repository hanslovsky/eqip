from __future__ import print_function
import glob

import logging
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

from fuse import ElasticAugment, Misalign, SimpleAugment, Snapshot, DefectAugment
from gunpowder import ArrayKey, Hdf5Source, Coordinate, BatchRequest, Normalize, Pad, RandomLocation, Reject, \
    ArraySpec, IntensityAugment, RandomProvider, GrowBoundary, IntensityScaleShift, \
    PrintProfilingStats, build, PreCache
from gunpowder.contrib import \
    ZeroOutConstSections
from gunpowder.ext import malis
from gunpowder.nodes import \
    AddAffinities,\
    RenumberConnectedComponents,\
    BalanceLabels
from gunpowder.tensorflow import Train
import tensorflow as tf
import os
import math
import json

from .. import gunpowder_utils
from .. import io_keys

RAW_KEY              = ArrayKey('RAW')
ALPHA_MASK_KEY       = ArrayKey('ALPHA_MASK')
GT_LABELS_KEY        = ArrayKey('GT_LABELS')
GT_MASK_KEY          = ArrayKey('GT_MASK')
TRAINING_MASK_KEY    = ArrayKey('TRAINING_MASK')
LOSS_GRADIENT_KEY    = ArrayKey('LOSS_GRADIENT')
AFFINITIES_KEY       = ArrayKey('AFFINITIES')
GT_AFFINITIES_KEY    = ArrayKey('GT_AFFINITIES')
AFFINITIES_MASK_KEY  = ArrayKey('AFFINITIES_MASK')
AFFINITIES_SCALE_KEY = ArrayKey('AFFINITIES_SCALE')

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
        summary,
        save_checkpoint_every,
        pre_cache_size,
        pre_cache_num_workers,
        snapshot_every,
        balance_labels,
        renumber_connected_components,
        network_inputs,
        ignore_labels_for_slip,
        grow_boundaries):

    ignore_keys_for_slip = (GT_LABELS_KEY, GT_MASK_KEY) if ignore_labels_for_slip else ()
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
    # specifiy which Arrays should be requested for each batch
    request = BatchRequest()
    request.add(RAW_KEY,             input_size,  voxel_size=input_voxel_size)
    request.add(GT_LABELS_KEY,       output_size, voxel_size=output_voxel_size)
    request.add(GT_AFFINITIES_KEY,   output_size, voxel_size=output_voxel_size)
    request.add(AFFINITIES_MASK_KEY, output_size, voxel_size=output_voxel_size)
    request.add(GT_MASK_KEY,         output_size, voxel_size=output_voxel_size)
    if balance_labels:
        request.add(AFFINITIES_SCALE_KEY, output_size, voxel_size=output_voxel_size)
    network_inputs[tensor_affinities_mask] = AFFINITIES_SCALE_KEY if balance_labels else AFFINITIES_MASK_KEY

    # create a tuple of data sources, one for each HDF file
    data_sources = tuple(
        provider +
        Normalize(RAW_KEY) + # ensures RAW is in float in [0, 1]

        # zero-pad provided RAW and GT_MASK to be able to draw batches close to
        # the boundary of the available data
        # size more or less irrelevant as followed by Reject Node
        Pad(RAW_KEY, None) +
        Pad(GT_MASK_KEY, None) +
        RandomLocation() + # chose a random location inside the provided arrays
        Reject(GT_MASK_KEY) + # reject batches wich do contain less than 50% labelled data
        Reject(GT_LABELS_KEY, min_masked=0.0, reject_probability=0.95)

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
                ALPHA_MASK_KEY : 'defect_sections/mask',
            },
            array_specs={
                RAW_KEY        : ArraySpec(voxel_size=input_voxel_size),
                ALPHA_MASK_KEY : ArraySpec(voxel_size=input_voxel_size),
            }
        ) +
        RandomLocation(min_masked=0.05, mask=ALPHA_MASK_KEY) +
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
    train_pipeline += Misalign(z_resolution=360, prob_slip=0.05, prob_shift=0.05, max_misalign=(360,) * 2, ignore_keys_for_slip=ignore_keys_for_slip)
    train_pipeline += SimpleAugment(transpose_only=[1,2])
    train_pipeline += IntensityAugment(RAW_KEY, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)
    train_pipeline += DefectAugment(RAW_KEY,
                      prob_missing=0.03,
                      prob_low_contrast=0.01,
                      prob_artifact=0.03,
                      artifact_source=artifact_source,
                      artifacts=RAW_KEY,
                      artifacts_mask=ALPHA_MASK_KEY,
                      contrast_scale=0.5)
    train_pipeline += IntensityScaleShift(RAW_KEY, 2, -1)
    train_pipeline += ZeroOutConstSections(RAW_KEY)
    if grow_boundaries > 0:
        train_pipeline += GrowBoundary(GT_LABELS_KEY, GT_MASK_KEY, steps=grow_boundaries, only_xy=True)

    if renumber_connected_components:
        train_pipeline += RenumberConnectedComponents(labels=GT_LABELS_KEY)

    train_pipeline += AddAffinities(
            affinity_neighborhood=affinity_neighborhood,
            labels=GT_LABELS_KEY,
            labels_mask=GT_MASK_KEY,
            affinities=GT_AFFINITIES_KEY,
            affinities_mask=AFFINITIES_MASK_KEY
        )

    if balance_labels:
        train_pipeline += BalanceLabels(labels=GT_AFFINITIES_KEY, scales=AFFINITIES_SCALE_KEY, mask=AFFINITIES_MASK_KEY)

    train_pipeline += PreCache(cache_size=pre_cache_size, num_workers=pre_cache_num_workers)
    train_pipeline += Train(
            summary=summary,
            graph=meta_graph_filename,
            save_every=save_checkpoint_every,
            optimizer=optimizer,
            loss=loss,
            inputs=network_inputs,
            log_dir='log',
            outputs={tensor_affinities: AFFINITIES_KEY},
            gradients={tensor_affinities: LOSS_GRADIENT_KEY},
            array_specs={
                AFFINITIES_KEY       : ArraySpec(voxel_size=output_voxel_size),
                LOSS_GRADIENT_KEY    : ArraySpec(voxel_size=output_voxel_size),
                AFFINITIES_MASK_KEY  : ArraySpec(voxel_size=output_voxel_size),
                GT_MASK_KEY          : ArraySpec(voxel_size=output_voxel_size),
                AFFINITIES_SCALE_KEY : ArraySpec(voxel_size=output_voxel_size)
            }
        )
    train_pipeline += Snapshot({
                RAW_KEY             : 'volumes/raw',
                GT_LABELS_KEY       : 'volumes/labels/neuron_ids',
                GT_AFFINITIES_KEY   : 'volumes/affinities/gt',
                AFFINITIES_KEY      : 'volumes/affinities/prediction',
                LOSS_GRADIENT_KEY   : 'volumes/loss_gradient',
                AFFINITIES_MASK_KEY : 'masks/affinities'
            },
            every=snapshot_every,
            output_filename='batch_{iteration}.hdf',
            output_dir='snapshots/',
            additional_request=snapshot_request)
    train_pipeline += PrintProfilingStats(every=50)

    print("Starting training...")
    with build(train_pipeline) as b:
        for i in range(trained_until, stop):
            b.request_batch(request)

    print("Training finished")


def train():

    import argparse

    def bounded_integer(val, lower=None, upper=None):
        val = int(val)
        if lower is not None and val < lower or upper is not None and val > upper:
            raise argparse.ArgumentTypeError('Value %d is out of bounds for [%s, %s]' % (val, str(-math.inf if lower is None else lower), str(math.inf if upper is None else upper)))
        return val

    def make_default_data_provider_string():
        data_dir = '/groups/saalfeld/home/hanslovskyp/experiments/quasi-isotropic/data/realigned'
        file_pattern = '*merged*fixed-offset-fixed-mask.h5'
        return '{}/{}:{}={}:{}={}:{}={}'.format(
            data_dir,
            file_pattern,
            'RAW', gunpowder_utils.DEFAULT_PATHS['raw'],
            'LABELS', gunpowder_utils.DEFAULT_PATHS['labels'],
            'MASK', gunpowder_utils.DEFAULT_PATHS['mask'])

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
    parser.add_argument('--pre-cache-num-workers', type=lambda arg: bounded_integer(arg, 1), default=50, metavar='PRECACHE_NUM_WORKERS', help='Number of workers used to populate pre-cache')
    parser.add_argument('--pre-cache-size', type=lambda arg: bounded_integer(arg, 1), default=100, metavar='PRECACHE_SIZE', help='Size of pre-cache')
    parser.add_argument('--ignore-labels-for-slip', action='store_true')
    parser.add_argument('--affinity-neighborhood-x', nargs='+', type=int, default=(-1,))
    parser.add_argument('--affinity-neighborhood-y', nargs='+', type=int, default=(-1,))
    parser.add_argument('--affinity-neighborhood-z', nargs='+', type=int, default=(-1,))
    parser.add_argument('--grow-boundaries', metavar='STEPS', type=int, default=0, help='Grow boundaries for STEPS if larger than 0.')
    parser.add_argument('--data-provider', metavar='GLOB_PATTERN[:raw=<dataset>[:labels=<dataset>:[mask=<dataset>]]]', default=(default_data_provider_string,), nargs='+', help='Add data provider.')

    args = parser.parse_args()
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
    data_providers = gunpowder_utils.make_data_providers(*args.data_provider)
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
        net_io_names[io_keys.GT_LABELS]     : GT_LABELS_KEY
    }

    if trained_until < args.mse_iterations:

        train_until(
            data_providers=data_providers,
            affinity_neighborhood=neighborhood,
            meta_graph_filename=args.meta_graph_filename,
            stop=args.mse_iterations - trained_until,
            input_shape=args.input_shape,
            output_shape=args.output_shape,
            loss=net_io_names['%s_%s' % (io_keys.MSE_PREFIX, io_keys.LOSS)],
            optimizer=net_io_names['%s_%s' % (io_keys.MSE_PREFIX, io_keys.OPTIMIZIER)],
            summary=net_io_names[io_keys.SUMMARY],
            tensor_affinities=net_io_names[io_keys.AFFINITIES],
            tensor_affinities_mask=net_io_names[io_keys.AFFINITIES_MASK],
            save_checkpoint_every=args.save_checkpoint_every,
            pre_cache_size=args.pre_cache_size,
            pre_cache_num_workers=args.pre_cache_num_workers,
            snapshot_every=args.snapshot_every,
            balance_labels=True,
            renumber_connected_components=False,
            network_inputs=inputs,
            ignore_labels_for_slip=args.ignore_labels_for_slip,
            grow_boundaries=args.grow_boundaries)

    if tf.train.latest_checkpoint('.') and int(tf.train.latest_checkpoint('.').split('_')[-1]) < args.mse_iterations:
        raise Exception("Inconsistency!")

    def malis_loss(graph):
        affinities = graph.get_tensor_by_name(net_io_names[io_keys.AFFINITIES])
        gt_affinities = graph.get_tensor_by_name(net_io_names[io_keys.GT_AFFINITIES])
        gt_labels = graph.get_tensor_by_name(net_io_names[io_keys.GT_LABELS])
        affinities_mask = graph.get_tensor_by_name(net_io_names[io_keys.AFFINITIES_MASK])
        loss = malis.malis_loss_op(
            affs = affinities,
            gt_affs = gt_affinities,
            gt_seg = gt_labels,
            neighborhood = neighborhood,
            gt_aff_mask = affinities_mask)
        opt = tf.train.AdamOptimizer(
                    learning_rate = 0.5e-4,
                    beta1         = 0.95,
                    beta2         = 0.999,
                    epsilon       = 1e-8,
                    name          = 'malis_adam_optimizer')
        optimizer = opt.minimize(loss)

        tf.summary.scalar('summary_malis_loss', loss)

        return loss, optimizer

    train_until(
        data_providers=data_providers,
        affinity_neighborhood=neighborhood,
        meta_graph_filename=args.meta_graph_filename,
        stop=args.malis_iterations - (trained_until - args.mse_iterations),
        input_shape=args.input_shape,
        output_shape=args.output_shape,
        loss=None,
        optimizer=malis_loss,
        summary='summary_malis_loss:0',
        tensor_affinities=net_io_names[io_keys.AFFINITIES],
        tensor_affinities_mask=net_io_names[io_keys.AFFINITIES_MASK],
        save_checkpoint_every=args.save_checkpoint_every,
        pre_cache_size=args.pre_cache_size,
        pre_cache_num_workers=args.pre_cache_num_workers,
        snapshot_every=args.snapshot_every,
        balance_labels=False,
        renumber_connected_components=True,
        network_inputs=inputs,
        ignore_labels_for_slip=args.ignore_labels_for_slip,
        grow_boundaries=args.grow_boundaries)
