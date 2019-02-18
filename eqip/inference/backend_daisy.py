from __future__ import division, print_function

import logging
import os

import daisy
from daisy import ClientScheduler, Roi

logging.basicConfig(level=logging.INFO)

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

import numpy as np

import z5py




def _default_pipeline_factory(
        input_container,
        input_dataset,
        output_filename,
        output_dir,
        output_dataset,
        output_compression_type,
        weight_graph,
        meta_graph,
        input_placeholder_tensor,
        output_placeholder_tensor,
        input_voxel_size,
        output_voxel_size):
    def factory():
        from gunpowder import ArrayKey, Pad, Normalize, IntensityScaleShift, ArraySpec
        from gunpowder.tensorflow import Predict

        from gunpowder.nodes.hdf5like_source_base import Hdf5LikeSource
        from gunpowder.nodes.hdf5like_write_base import Hdf5LikeWrite
        from gunpowder.coordinate import Coordinate
        from gunpowder.compat import ensure_str

        from fuse import Z5Source, Z5Write

        _RAW = ArrayKey('RAW')
        _AFFS = ArrayKey('AFFS')

        input_source = Z5Source(
            input_container,
            datasets={_RAW: input_dataset},
            array_specs={_RAW: ArraySpec(voxel_size=input_voxel_size)})
        output_write = Z5Write(
            output_filename=output_filename,
            output_dir=output_dir,
            dataset_names={_AFFS: output_dataset},
            compression_type=output_compression_type)
        return \
            input_source + \
            Normalize(_RAW) + \
            Pad(_RAW, size=None) + \
            IntensityScaleShift(_RAW, 2, -1) + \
            Predict(
                weight_graph,
                inputs={
                    input_placeholder_tensor: _RAW
                },
                outputs={
                    output_placeholder_tensor: _AFFS
                },
                graph=meta_graph,
                array_specs={_AFFS: ArraySpec(voxel_size=output_voxel_size)}
            ) + \
            output_write
    return factory

def make_process_function(
        actor_id_to_gpu_mapping,
        pipeline_factory,
        input_voxel_size,
        output_voxel_size):
    def process_function():
        scheduler = ClientScheduler()
        actor_id = scheduler.context.actor_id
        num_workers = scheduler.context.num_workers
        gpu = actor_id_to_gpu_mapping(actor_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        _logger.info("Worker %d uses gpu %d", actor_id, gpu)

        _logger.info("Environment:")
        for name in os.environ.keys():
            _logger.info('  %s:%s', name, os.environ[name])

        # _logger.info("GPU is available: %s", tf.test.is_gpu_available())

        # from tensorflow.python.client import device_lib

        # def get_available_gpus():
            # local_device_protos = device_lib.list_local_devices()
            # return [x.name for x in local_device_protos if x.device_type == 'GPU']

        # available_gpus = get_available_gpus()
        # for gpu in available_gpus:
            # print("Worker %d  sees gpus %s" % (actor_id, available_gpus))

        import tensorflow as tf
        with tf.device('/gpu:%d' % 0):
            from gunpowder import ArrayKey, ArraySpec, build, BatchRequest
            from gunpowder import Roi as gRoi
            from gunpowder import Coordinate as gCoordinate
            _RAW = ArrayKey('RAW')
            _AFFS = ArrayKey('AFFS')

            num_predicted_blocks = 0
            pipeline = pipeline_factory()
            with build(pipeline):
                while True:
                    block = scheduler.acquire_block()
                    if block is None:
                        break

                    request = BatchRequest()
                    request[_RAW] = ArraySpec(
                        roi=gRoi(offset=block.read_roi.get_begin(), shape=block.read_roi.get_shape()),
                        voxel_size=gCoordinate(input_voxel_size))
                    request[_AFFS] = ArraySpec(
                        roi=gRoi(offset=block.write_roi.get_begin(), shape=block.write_roi.get_shape()),
                        voxel_size=gCoordinate(output_voxel_size))
                    _logger.info('Requesting %s', request)
                    pipeline.request_batch(request)
                    scheduler.release_block(block, 0)
                    num_predicted_blocks += 1
                    _logger.info("Worker %d predicted %d blocks", actor_id, num_predicted_blocks)

    return process_function

def predict_affinities_daisy():
    import timeit
    start = timeit.default_timer()
    _predict_affinities_daisy()
    stop = timeit.default_timer()
    _logger.info("Wall clock time for inference: %ds", stop - start)

def _predict_affinities_daisy():



    import pathlib
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-container', type=str, required=True, help='N5 container')
    parser.add_argument('--input-dataset', type=str, required=True, help='3-dimensional')
    parser.add_argument('--output-container', type=str, required=True, help='N5 container')
    parser.add_argument('--output-dataset', type=str)
    parser.add_argument('--num-channels', required=True, type=int)
    parser.add_argument('--gpus', required=True, type=int, nargs='+')
    parser.add_argument('--input-voxel-size', nargs=3, type=int, default=(360, 36, 36), help='zyx')
    parser.add_argument('--output-voxel-size', nargs=3, type=int, default=(120, 108, 108), help='zyx')
    parser.add_argument('--network-input-shape', nargs=3, type=int, default=(91, 862, 862), help='zyx')
    parser.add_argument('--network-output-shape', nargs=3, type=int, default=(207, 214, 214), help='zyx')
    parser.add_argument('--experiment-directory', required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--weight-graph-pattern', default='unet_checkpoint_%d', help='Relative to experiment-directory.')
    parser.add_argument('--meta-graph-filename', default='unet-inference.meta', help='Relative to experiment-directory.')
    parser.add_argument('--input-placeholder-tensor', default='Placeholder:0')
    parser.add_argument('--output-placeholder-tensor', default='Slice:0')
    parser.add_argument('--output-compression', default='raw')

    args = parser.parse_args()

    input_voxel_size = np.array(args.input_voxel_size, dtype=np.float64)
    output_voxel_size = np.array(args.output_voxel_size, dtype=np.float64)

    experiment_directory = args.experiment_directory
    input_container = args.input_container
    input_dataset = args.input_dataset
    output_container = pathlib.Path(args.output_container)
    output_dir = output_container.parent
    output_dataset = args.output_dataset
    iteration = args.iteration
    network_input_shape = np.array(args.network_input_shape, dtype=np.uint64)
    network_input_shape_world = np.array(tuple(n * i for n, i in zip(network_input_shape, input_voxel_size)), dtype=np.float64)
    network_output_shape = np.array(args.network_output_shape, dtype=np.uint64)
    network_output_shape_world = np.array(tuple(n * o for n, o in zip(network_output_shape, output_voxel_size)), dtype=np.float64)
    shape_diff_world = network_input_shape_world - network_output_shape_world
    input_placeholder_tensor = args.input_placeholder_tensor
    output_placeholder_tensor = args.output_placeholder_tensor

    with z5py.File(path=input_container, use_zarr_format=False, mode='r') as f:
        ds = f[input_dataset]
        input_dataset_size = ds.shape
    input_dataset_size_world  = np.array(tuple(vs * s for vs, s in zip(input_voxel_size, input_dataset_size)), dtype=np.float64)
    output_dataset_roi_world = Roi(
        shape=input_dataset_size_world,
        offset = np.array((0,) * len(input_dataset_size_world), dtype=np.float64))
    output_dataset_roi_world = output_dataset_roi_world.snap_to_grid(network_output_shape_world, mode='grow')
    output_dataset_roi = output_dataset_roi_world / tuple(output_voxel_size)

    num_channels = args.num_channels

    _logger.info('input dataset size world:   %s', input_dataset_size_world)
    _logger.info('output dataset roi world:   %s', output_dataset_roi_world)
    _logger.info('output datset roi:          %s', output_dataset_roi)
    _logger.info('output network size:        %s', network_output_shape)
    _logger.info('output network size world:  %s', network_output_shape_world)


    if not os.path.isdir(str(output_container)):
        os.makedirs(str(output_container))
    with z5py.File(str(output_container), use_zarr_format=False) as f:
        ds = f.require_dataset(
            name=output_dataset,
            shape=(num_channels,) + tuple(int(s) for s in output_dataset_roi.get_shape()) if num_channels > 0 else tuple(int(s) for s in output_dataset_roi.get_shape()),
            dtype=np.float32,
            chunks = (1,) + tuple(int(n) for n in network_output_shape) if num_channels > 0 else tuple(int(n) for n in network_output_shape),
            compression='raw')
        ds.attrs['resolution'] = tuple(args.output_voxel_size[::-1])
        ds.attrs['offset'] = tuple(output_dataset_roi_world.get_begin()[::-1])

    gpus = args.gpus
    num_workers = len(gpus)

    pipeline_factory = _default_pipeline_factory(
        input_container=input_container,
        input_dataset=input_dataset,
        output_filename=str(output_container.name),
        output_dir=str(output_dir),
        output_dataset=output_dataset,
        output_compression_type=args.output_compression,
        weight_graph=os.path.join(experiment_directory, args.weight_graph_pattern % iteration),
        meta_graph=os.path.join(experiment_directory, args.meta_graph_filename),
        input_placeholder_tensor=input_placeholder_tensor,
        output_placeholder_tensor=output_placeholder_tensor,
        input_voxel_size=input_voxel_size,
        output_voxel_size=output_voxel_size)

    process_function = make_process_function(
        actor_id_to_gpu_mapping=lambda id: gpus[id],
        pipeline_factory=pipeline_factory,
        input_voxel_size=input_voxel_size,
        output_voxel_size=output_voxel_size)

    total_roi = output_dataset_roi_world.grow(amount_neg=tuple(shape_diff_world / 2), amount_pos=tuple(shape_diff_world / 2))
    read_roi  = Roi(shape=tuple(network_input_shape_world), offset=tuple(-shape_diff_world / 2))
    write_roi = Roi(shape=tuple(network_output_shape_world), offset=tuple(np.array((0,) * len(input_voxel_size))))
    _logger.info('Running blockwise!')
    _logger.info('total roi:   %s', total_roi)
    _logger.info('read  roi:   %s', read_roi)
    _logger.info('write roi:   %s', write_roi)
    daisy.run_blockwise(
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=process_function,
        num_workers=num_workers,
        read_write_conflict=False)

if __name__ == '__main__':
    predict_affinities_daisy()
