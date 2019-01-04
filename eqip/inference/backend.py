from __future__ import print_function, division

import logging
logging.basicConfig()
logging.getLogger('gunpowder.nodes.scan').setLevel(logging.INFO)
logging.getLogger('gunpowder.nodes.pad').setLevel(logging.INFO)
logging.getLogger('gunpowder.nodes.hdf5like_source_base').setLevel(logging.INFO)
logging.getLogger('gunpowder.nodes.hdf5like_write_base').setLevel(logging.DEBUG)
logging.getLogger('gunpowder.nodes.batch_provider').setLevel(logging.INFO)
logging.getLogger('gunpowder.tensorflow.nodes.predict').setLevel(logging.INFO)

import numpy as np

import z5py

from gunpowder import ArrayKey, BatchRequest, Pad, Crop, Normalize, IntensityScaleShift, Scan, build, Coordinate, \
    ArraySpec, Roi
from gunpowder.tensorflow import Predict

_RAW = ArrayKey('RAW')
_AFFS = ArrayKey('AFFS')


def predict(
        experiment_directory,
        iteration,
        input_source,
        output_write,
        output_roi_world,
        input_chunk_size_world,
        output_chunk_size_world,
        input_placeholder_tensor,
        output_placeholder_tensor,
        input_voxel_size=Coordinate((360, 36, 36)),
        output_voxel_size=Coordinate((120, 108, 108)),
        graph_meta='unet-inference.meta',
        weight_graph_pattern='unet_checkpoint_%d',
        RAW=_RAW,
        AFFS=_AFFS):

    chunk_request = BatchRequest()
    chunk_request.add(AFFS, shape=output_chunk_size_world, voxel_size=output_voxel_size)
    chunk_request.add(RAW, shape=input_chunk_size_world, voxel_size=input_voxel_size)

    pipeline = (
        input_source +
        Normalize(RAW) +
        Pad(RAW, size=None) +
        IntensityScaleShift(RAW, 2, -1) +
        Predict(
            os.path.join(experiment_directory, weight_graph_pattern % iteration),
            inputs={
                input_placeholder_tensor: RAW
            },
            outputs={
                output_placeholder_tensor: AFFS
            },
            graph=os.path.join(experiment_directory, graph_meta),
            array_specs={_AFFS: ArraySpec(voxel_size=output_voxel_size)}
        ) +
        output_write +
        Scan(chunk_request, num_workers=1)
    )

    print("Starting prediction...")
    scan_request = BatchRequest()
    scan_request[AFFS] = ArraySpec(roi=output_roi_world, voxel_size=output_voxel_size)
    # scan_request[AFFS] = ArraySpec(roi=output_roi_world.shift(chunk_request[AFFS].roi.get_begin()), voxel_size=output_voxel_size)
    # scan_request.add(AFFS, shape=output_roi_world.shift(chunk_request[AFFS].roi.get_begin()), voxel_size=output_voxel_size)
    with build(pipeline):
        pipeline.request_batch(scan_request)
    print("Prediction finished")

if __name__ == "__main__":

    from gunpowder.compat import ensure_str
    from gunpowder.coordinate import Coordinate, Coordinate, Coordinate
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
    import logging
    import os
    import traceback

    logger = logging.getLogger(__name__)

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


    input_voxel_size = Coordinate((360, 36, 36))
    output_voxel_size = Coordinate((120, 108, 108))

    experiment_directory = '/groups/saalfeld/home/hanslovskyp/experiments/quasi-isotropic/15/'
    input_container = '/groups/saalfeld/home/hanslovskyp/data/cremi/sample_A+_padded_20160601-bs=64.n5'
    input_dataset = 'volumes/raw/data/s0'
    output_dir = '/groups/saalfeld/home/hanslovskyp/data/cremi'
    output_container = 'prediction-full.n5'
    output_dataset = 'volumes/affinities/predictions'
    input_source = Z5Source(input_container, datasets={_RAW: input_dataset}, array_specs={_RAW: ArraySpec(voxel_size=input_voxel_size)})
    output_write = Z5Write(
        output_filename=output_container,
        output_dir=output_dir,
        dataset_names={_AFFS: output_dataset},
        compression_type='raw')
    iteration = 790000
    network_input_shape = Coordinate((91, 862, 862))
    # network_input_shape = Coordinate((43, 430, 430))
    network_input_shape_world = Coordinate(tuple(n * i for n, i in zip(network_input_shape, input_voxel_size)))
    network_output_shape = Coordinate((209, 214, 214))
    # network_output_shape = Coordinate((65, 70, 70))
    network_output_shape_world = Coordinate(tuple(n * o for n, o in zip(network_output_shape, output_voxel_size)))
    shape_diff = network_input_shape_world - network_output_shape_world
    roi = tuple(f * s for f, s in zip((3, 3, 3), network_output_shape))
    roi_world = Coordinate(tuple(r * o for r, o in zip(roi, output_voxel_size)))
    input_placeholder_tensor="Placeholder:0",
    output_placeholder_tensor="Reshape_1:0"

    with z5py.File(path=input_container, use_zarr_format=False, mode='r') as f:
        ds = f[input_dataset]
        input_dataset_size = ds.shape
    input_dataset_size_world  = Coordinate(tuple(vs * s for vs, s in zip(input_voxel_size, input_dataset_size)))
    output_dataset_size_world = input_dataset_size_world - shape_diff
    output_dataset_roi_world  = Roi(shape=output_dataset_size_world, offset=shape_diff / 2).snap_to_grid(output_voxel_size, mode='shrink')
    output_dataset_roi        = output_dataset_roi_world / output_voxel_size

    print("LOL?", output_dataset_size_world, input_dataset_size_world)

    with z5py.File(path=os.path.join(output_dir, output_container), use_zarr_format=False) as f:
        ds = f.require_dataset(
            name=output_dataset,
            shape=(3,) + output_dataset_roi.get_shape(),
            dtype=np.float32,
            chunks = (3,) + tuple(s for s in network_output_shape),
            compression='raw'
        )
        ds.attrs['resolution'] = (120.0, 108.0, 108.0)
        ds.attrs['offset'] = tuple(s for s in output_dataset_roi_world.get_begin()) #tuple(sd / 2 for sd in shape_diff)

    print("Predicting for roi", output_dataset_roi_world, output_dataset_roi)
    print("shape diff", shape_diff)


    predict(
        experiment_directory=experiment_directory,
        iteration=iteration,
        input_source=input_source,
        output_write=output_write,
        output_roi_world=output_dataset_roi_world,#Roi(shape=roi_world, offset=(0,) * len(roi_world)),
        input_chunk_size_world=network_input_shape_world,
        output_chunk_size_world=network_output_shape_world,
        input_placeholder_tensor=input_placeholder_tensor,
        output_placeholder_tensor=output_placeholder_tensor,
    )
