from __future__ import division, print_function

import json
import logging
import os

import daisy
from daisy import Client, Roi

logging.basicConfig(level=logging.INFO)

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

import numpy as np

import z5py




def _default_pipeline_factory(
        input_container,
        input,
        output_filename,
        output_dir,
        output_compression_type,
        outputs,
        weight_graph,
        meta_graph,
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

        output_dataset_names = {ArrayKey('OUTPUT_%d' % i): ds for i, (ds, _) in enumerate(outputs)}
        output_tensor_to_key = {tensor: ArrayKey('OUTPUT_%d' % i) for i, (_, tensor) in enumerate(outputs)}
        output_array_specs   = {ArrayKey('OUTPUT_%d' % i): ArraySpec(voxel_size=output_voxel_size) for i in range(len(outputs))}


        input_source = Z5Source(
            input_container,
            datasets={_RAW: input[0]},
            array_specs={_RAW: ArraySpec(voxel_size=input_voxel_size)})
        output_write = Z5Write(
            output_filename=output_filename,
            output_dir=output_dir,
            dataset_names=output_dataset_names,
            compression_type=output_compression_type)
        return \
            input_source + \
            Normalize(_RAW) + \
            Pad(_RAW, size=None) + \
            IntensityScaleShift(_RAW, 2, -1) + \
            Predict(
                weight_graph,
                inputs={input[1]: _RAW},
                outputs=output_tensor_to_key,
                graph=meta_graph,
                array_specs=output_array_specs) + \
            output_write
    return factory

def make_process_function(
        actor_id_to_gpu_mapping,
        pipeline_factory,
        input_voxel_size,
        output_voxel_size,
        outputs,
        num_cpu_workers):
    def process_function():
        scheduler = Client()
        worker_id = scheduler.context.worker_id
        num_workers = scheduler.context.num_workers
        gpu = actor_id_to_gpu_mapping(worker_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        _logger.info("Worker %d uses gpu %d with %d workers", worker_id, gpu, num_workers)

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
            from gunpowder import ArrayKey, ArraySpec, build, BatchRequest, DaisyRequestBlocks
            _RAW = ArrayKey('RAW')

            roi_map = {ArrayKey('OUTPUT_%d' % i): 'write_roi' for i in range(len(outputs))}
            roi_map[_RAW] = 'read_roi'

            reference = BatchRequest()
            reference[_RAW] = ArraySpec(roi=None, voxel_size=input_voxel_size)
            for i in range(len(outputs)):
                reference[ArrayKey('OUTPUT_%d' % i)] = ArraySpec(roi=None, voxel_size=output_voxel_size)

            pipeline = pipeline_factory()
            pipeline += DaisyRequestBlocks(reference=reference, roi_map=roi_map, num_workers=num_cpu_workers)
            with build(pipeline):
                pipeline.request_batch(BatchRequest())


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
    # parser.add_argument('--input-dataset', type=str, required=True, help='3-dimensional')
    parser.add_argument('--output-container', type=str, required=True, help='N5 container')
    parser.add_argument('--input', type=str, nargs=2, metavar=('dataset', 'tensor'), help='For example --input volumes/raw Placeholder:0', required=True)
    parser.add_argument('--output', type=str, action='append', nargs=4, metavar=('dataset', 'dtype', 'num_channels', 'tensor'), help='For example --output volumes/affinities/prediction float32 3 Slice:0. num-channels<=0 means no channel axis', required=True)
    # parser.add_argument('--output-dataset', type=str)
    parser.add_argument('--gpus', required=True, type=int, nargs='+')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of CPU workers per GPU for parallel processing')
    parser.add_argument('--input-voxel-size', nargs=3, type=int, default=(360, 36, 36), help='zyx')
    parser.add_argument('--output-voxel-size', nargs=3, type=int, default=(120, 108, 108), help='zyx')
    parser.add_argument('--network-input-shape', nargs=3, type=int, default=(91, 862, 862), help='zyx')
    parser.add_argument('--network-output-shape', nargs=3, type=int, default=(207, 214, 214), help='zyx')
    parser.add_argument('--experiment-directory', required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--weight-graph-pattern', default='unet_checkpoint_%d', help='Relative to experiment-directory.')
    parser.add_argument('--meta-graph-filename', default='unet-inference.meta', help='Relative to experiment-directory.')
    # parser.add_argument('--input-placeholder-tensor', default='Placeholder:0')
    # parser.add_argument('--output-placeholder-tensor', default='Slice:0')
    parser.add_argument('--output-compression', default='raw')
    parser.add_argument('--net-io-names', default=None, required=False, help='Look-up tensor names from json, if specified. Use specified values from --input/--output as tensor names directly, otherwise.')

    args = parser.parse_args()

    input_voxel_size = np.array(args.input_voxel_size, dtype=np.float64)
    output_voxel_size = np.array(args.output_voxel_size, dtype=np.float64)

    if args.net_io_names:
        with open(args.net_io_names, 'r') as f:
            net_io_names = json.load(f)
        def tensor_name(name):
            return net_io_names[name]
    else:
        def tensor_name(name):
            return name


    experiment_directory = args.experiment_directory
    input_container = args.input_container
    output_container = pathlib.Path(args.output_container)
    output_dir = output_container.parent
    iteration = args.iteration
    network_input_shape = np.array(args.network_input_shape, dtype=np.uint64)
    network_input_shape_world = np.array(tuple(n * i for n, i in zip(network_input_shape, input_voxel_size)), dtype=np.float64)
    network_output_shape = np.array(args.network_output_shape, dtype=np.uint64)
    network_output_shape_world = np.array(tuple(n * o for n, o in zip(network_output_shape, output_voxel_size)), dtype=np.float64)
    shape_diff_world = network_input_shape_world - network_output_shape_world
    print(args.input, args.output)
    inputs =  tuple((ds, tensor_name(tensor)) for ds, tensor in [args.input])
    outputs = tuple((ds, np.dtype(dtype), int(nc), tensor_name(tensor)) for ds, dtype, nc, tensor in args.output)

    with z5py.File(path=input_container, use_zarr_format=False, mode='r') as f:
        ds = f[inputs[0][0]]
        input_dataset_size = ds.shape
    input_dataset_size_world  = np.array(tuple(vs * s for vs, s in zip(input_voxel_size, input_dataset_size)), dtype=np.float64)
    output_dataset_roi_world = Roi(
        shape=input_dataset_size_world,
        offset = np.array((0,) * len(input_dataset_size_world), dtype=np.float64))
    output_dataset_roi_world = output_dataset_roi_world.snap_to_grid(network_output_shape_world, mode='grow')
    output_dataset_roi = output_dataset_roi_world / tuple(output_voxel_size)

    _logger.info('input dataset size world:   %s', input_dataset_size_world)
    _logger.info('output dataset roi world:   %s', output_dataset_roi_world)
    _logger.info('output datset roi:          %s', output_dataset_roi)
    _logger.info('output network size:        %s', network_output_shape)
    _logger.info('output network size world:  %s', network_output_shape_world)


    weight_graph = os.path.join(experiment_directory, args.weight_graph_pattern % iteration)
    meta_graph = os.path.join(experiment_directory, args.meta_graph_filename)


    if not os.path.isdir(str(output_container)):
        os.makedirs(str(output_container))
    with z5py.File(str(output_container), use_zarr_format=False) as f:

        for output_dataset, dtype, num_channels, tensor in outputs:

            ds = f.require_dataset(
                name=output_dataset,
                shape=(num_channels,) + tuple(int(s) for s in output_dataset_roi.get_shape()) if num_channels > 0 else tuple(int(s) for s in output_dataset_roi.get_shape()),
                dtype=dtype,
                chunks = (1,) + tuple(int(n) for n in network_output_shape) if num_channels > 0 else tuple(int(n) for n in network_output_shape),
                compression='raw')
            ds.attrs['resolution'] = tuple(args.output_voxel_size[::-1])
            ds.attrs['offset'] = tuple(output_dataset_roi_world.get_begin()[::-1])
            workflow_info = {
                'input'  : {'container': input_container, 'dataset': inputs[0][0], 'tensor': inputs[0][1]},
                'output' : {'tensor': tensor},
                'network': {'experiment_directory': experiment_directory, 'weight_graph': weight_graph, 'meta_graph': meta_graph, 'iteration': iteration}
            }
            ds.attrs['workflow_info'] = workflow_info

    gpus = args.gpus
    num_workers = len(gpus)

    pipeline_factory = _default_pipeline_factory(
        input_container=input_container,
        input=inputs[0],
        output_filename=str(output_container.name),
        output_dir=str(output_dir),
        outputs=tuple((ds, tensor) for ds, _, _, tensor in outputs),
        output_compression_type=args.output_compression,
        weight_graph=weight_graph,
        meta_graph=meta_graph,
        input_voxel_size=input_voxel_size,
        output_voxel_size=output_voxel_size)

    process_function = make_process_function(
        actor_id_to_gpu_mapping=lambda id: gpus[id],
        pipeline_factory=pipeline_factory,
        input_voxel_size=input_voxel_size,
        output_voxel_size=output_voxel_size,
        outputs=tuple((ds, tensor) for ds, _, _, tensor in outputs),
        num_cpu_workers=args.num_workers)

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
