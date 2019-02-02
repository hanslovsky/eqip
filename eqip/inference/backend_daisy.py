from __future__ import print_function, division

import os

import logging

import daisy
from daisy import ClientScheduler

logging.basicConfig(level=logging.INFO)

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

import numpy as np

import z5py

from gunpowder import ArrayKey, BatchRequest, Pad, Normalize, IntensityScaleShift, build, \
    ArraySpec, Roi
from gunpowder.tensorflow import Predict

_RAW = ArrayKey('RAW')
_AFFS = ArrayKey('AFFS')

from tensorflow.python.client import device_lib

def _get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def _default_pipeline_factory(
        input_source,
        output_write,
        weight_graph,
        meta_graph,
        input_placeholder_tensor,
        output_placeholder_tensor,
        output_voxel_size,
        RAW,
        AFFS):
    return lambda: \
        input_source + \
        Normalize(RAW) + \
        Pad(RAW, size=None) + \
        IntensityScaleShift(RAW, 2, -1) + \
        Predict(
            weight_graph,
            inputs={
                input_placeholder_tensor: RAW
            },
            outputs={
                output_placeholder_tensor: AFFS
            },
            graph=meta_graph,
            array_specs={_AFFS: ArraySpec(voxel_size=output_voxel_size)}
        ) + \
        output_write

def make_process_function(
        actor_id_to_gpu_mapping,
        pipeline_factory,
        input_voxel_size,
        output_voxel_size,
        RAW=_RAW,
        AFFS=_AFFS):
    def process_function():
        scheduler = ClientScheduler()
        actor_id = scheduler.context.actor_id
        num_workers = scheduler.context.num_workers
        gpu = actor_id_to_gpu_mapping(actor_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        _logger.info("Worker %d uses gpu %d", actor_id, gpu)

        num_predicted_blocks = 0
        pipeline = pipeline_factory()
        with build(pipeline):
            while True:
                block = scheduler.acquire_block()
                if block is None:
                    break

                request = BatchRequest()
                request[RAW] = ArraySpec(roi=block.read_roi, voxel_size=input_voxel_size)
                request[AFFS] = ArraySpec(roi=block.write_roi, voxel_size=output_voxel_size)
                _logger.info('Requesting %s', request)
                pipeline.request_batch(request)
                scheduler.release_block(block, 0)
                num_predicted_blocks += 1
        _logger.info("Worker %d predicted %d blocks", actor_id, num_predicted_blocks)

    return process_function

def predict_affinities_daisy():

    from gunpowder.nodes.hdf5like_source_base import Hdf5LikeSource


    class Z5Source(Hdf5LikeSource):
        '''A `zarr <https://github.com/zarr-developers/zarr>`_ data source.

        Provides arrays from zarr datasets. If the attribute ``resolution`` is set
        in a zarr dataset, it will be used as the array's ``voxel_size``. If the
        attribute ``offset`` is set in a dataset, it will be used as the offset of
        the :class:`Roi` for this array. It is assumed that the offset is given in
        world units.

        Args:

            filename (``string``):

                The zarr directory.

            datasets (``dict``, :class:`ArrayKey` -> ``string``):

                Dictionary of array keys to dataset names that this source offers.

            array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`, optional):

                An optional dictionary of array keys to array specs to overwrite
                the array specs automatically determined from the data file. This
                is useful to set a missing ``voxel_size``, for example. Only fields
                that are not ``None`` in the given :class:`ArraySpec` will be used.
        '''

        def _get_array_attribute(self, dataset, attribute, fallback_value, revert=False):
            val = dataset.attrs[attribute] if attribute in dataset.attrs else [fallback_value] * 3
            return val[::-1] if revert else val

        def _revert(self):
            return self.filename.endswith('.n5')

        def _get_voxel_size(self, dataset):
            return Coordinate(self._get_array_attribute(dataset, 'resolution', 1, revert=self._revert()))

        def _get_offset(self, dataset):
            return Coordinate(self._get_array_attribute(dataset, 'offset', 0, revert=self._revert()))

        def _open_file(self, filename):
            return z5py.File(ensure_str(filename), mode='r')

    from gunpowder.nodes.hdf5like_write_base import Hdf5LikeWrite
    from gunpowder.coordinate import Coordinate
    from gunpowder.compat import ensure_str

    class Z5Write(Hdf5LikeWrite):
        '''Assemble arrays of passing batches in one zarr container. This is useful
        to store chunks produced by :class:`Scan` on disk without keeping the
        larger array in memory. The ROIs of the passing arrays will be used to
        determine the position where to store the data in the dataset.

        Args:

            dataset_names (``dict``, :class:`ArrayKey` -> ``string``):

                A dictionary from array keys to names of the datasets to store them
                in.

            output_dir (``string``):

                The directory to save the zarr container. Will be created, if it does
                not exist.

            output_filename (``string``):

                The output filename of the container. Will be created, if it does
                not exist, otherwise data is overwritten in the existing container.

            compression_type (``string`` or ``int``):

                Compression strategy.  Legal values are ``gzip``, ``szip``,
                ``lzf``. If an integer between 1 and 10, this indicates ``gzip``
                compression level.

            dataset_dtypes (``dict``, :class:`ArrayKey` -> data type):

                A dictionary from array keys to datatype (eg. ``np.int8``). If
                given, arrays are stored using this type. The original arrays
                within the pipeline remain unchanged.
        '''

        def _get_array_attribute(self, dataset, attribute, fallback_value, revert=False):
            val = dataset.attrs[attribute] if attribute in dataset.attrs else [fallback_value] * 3#len(dataset.shape)
            return val[::-1] if revert else val

        def _revert(self):
            return os.path.join(self.output_dir, self.output_filename).endswith('.n5')

        def _get_voxel_size(self, dataset):
            return Coordinate(self._get_array_attribute(dataset, 'resolution', 1, revert=self._revert()))

        def _get_offset(self, dataset):
            return Coordinate(self._get_array_attribute(dataset, 'offset', 0, revert=self._revert()))

        def _set_voxel_size(self, dataset, voxel_size):

            if self.output_filename.endswith('.n5'):
                dataset.attrs['resolution'] = voxel_size[::-1]
            else:
                dataset.attrs['resolution'] = voxel_size

        def _set_offset(self, dataset, offset):

            if self.output_filename.endswith('.n5'):
                dataset.attrs['offset'] = offset[::-1]
            else:
                dataset.attrs['offset'] = offset

        def _open_file(self, filename):
            return z5py.File(ensure_str(filename), mode='a')

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

    input_voxel_size = Coordinate(args.input_voxel_size)
    output_voxel_size = Coordinate(args.output_voxel_size)

    experiment_directory = args.experiment_directory
    input_container = args.input_container
    input_dataset = args.input_dataset
    output_container = pathlib.Path(args.output_container)
    output_dir = output_container.parent
    output_dataset = args.output_dataset
    input_source = Z5Source(input_container, datasets={_RAW: input_dataset}, array_specs={_RAW: ArraySpec(voxel_size=input_voxel_size)})
    output_write = Z5Write(
        output_filename=str(output_container.name),
        output_dir=str(output_dir),
        dataset_names={_AFFS: output_dataset},
        compression_type=args.output_compression)
    iteration = args.iteration
    network_input_shape = Coordinate(args.network_input_shape)
    network_input_shape_world = Coordinate(tuple(n * i for n, i in zip(network_input_shape, input_voxel_size)))
    network_output_shape = Coordinate(args.network_output_shape)
    network_output_shape_world = Coordinate(tuple(n * o for n, o in zip(network_output_shape, output_voxel_size)))
    shape_diff_world = network_input_shape_world - network_output_shape_world
    input_placeholder_tensor = args.input_placeholder_tensor
    output_placeholder_tensor = args.output_placeholder_tensor

    with z5py.File(path=input_container, use_zarr_format=False, mode='r') as f:
        ds = f[input_dataset]
        input_dataset_size = ds.shape
    input_dataset_size_world  = Coordinate(tuple(vs * s for vs, s in zip(input_voxel_size, input_dataset_size)))
    output_dataset_roi_world = Roi(
        shape=input_dataset_size_world,
        offset = Coordinate((0,) * len(input_dataset_size_world)))
    output_dataset_roi_world = output_dataset_roi_world.snap_to_grid(network_output_shape_world, mode='grow')
    output_dataset_roi = output_dataset_roi_world / output_voxel_size

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
            shape=(num_channels,) + output_dataset_roi.get_shape() if num_channels > 0 else output_dataset_roi.get_shape(),
            dtype=np.float32,
            chunks = (1,) + tuple(network_output_shape) if num_channels > 0 else tuple(network_output_shape),
            compression='raw')
        ds.attrs['resolution'] = args.output_voxel_size[::-1]
        ds.attrs['offset'] = output_dataset_roi_world.get_begin()[::-1]

    input_key = _RAW
    output_key = _AFFS

    gpus = args.gpus
    num_workers = len(gpus)

    pipeline_factory = _default_pipeline_factory(
        input_source=input_source,
        output_write=output_write,
        weight_graph=os.path.join(experiment_directory, args.weight_graph_pattern % iteration),
        meta_graph=os.path.join(experiment_directory, args.meta_graph_filename),
        AFFS=output_key,
        RAW=input_key,
        input_placeholder_tensor=input_placeholder_tensor,
        output_placeholder_tensor=output_placeholder_tensor,
        output_voxel_size=output_voxel_size)

    process_function = make_process_function(
        actor_id_to_gpu_mapping=lambda id: gpus[id],
        pipeline_factory=pipeline_factory,
        input_voxel_size=input_voxel_size,
        output_voxel_size=output_voxel_size,
        RAW=input_key,
        AFFS=output_key)

    total_roi = output_dataset_roi_world.grow(amount_neg=shape_diff_world / 2, amount_pos=shape_diff_world / 2)
    read_roi  = Roi(shape=network_input_shape_world, offset=-shape_diff_world / 2)
    write_roi = Roi(shape=network_output_shape_world, offset=Coordinate((0,) * len(input_voxel_size)))
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
