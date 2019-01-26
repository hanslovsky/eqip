from __future__ import division

import os
import time
import json
from threading import Thread

import math
import z5py

from simpleference.inference.inference import run_inference_n5
from simpleference.backends.gunpowder.tensorflow.backend import TensorflowPredict
from simpleference.backends.gunpowder.preprocess import preprocess
from simpleference.inference.util import get_offset_lists


def gpu_inference(
        meta_graph_basename,
        weight_graph_basename,
        input_container,
        output_container,
        input_dataset,
        output_dataset,
        input_voxel_size,
        output_voxel_size,
        network_input_shape,
        network_output_shape,
        num_channels,
        gpus,
        offsets_dir):

    assert os.path.exists(input_container)
    voxel_size_ratios = tuple(i / o for i, o in zip(input_voxel_size, output_voxel_size))
    for i, o in zip(input_voxel_size, output_voxel_size):
        assert int(i / o) == i / o or int(o / i) == o / i, \
            'Input and output voxel size must be integer multiple (or vice versa) but got %s and %s' % (input_voxel_size, output_voxel_size)
    rf = z5py.File(input_container, use_zarr_format=False)
    input_shape = rf[input_dataset].shape
    output_shape = tuple(int(math.ceil(i * r)) for i, r in zip(input_shape, voxel_size_ratios)) + (num_channels,)

    print('input_shape         %s' % (input_shape,))
    print('output_shape        %s' % (output_shape,))
    print('input_voxel_size    %s' % (input_voxel_size,))
    print('output_voxel_size   %s' % (output_voxel_size,))


    # the n5 file might exist already
    # if not os.path.exists(output_container):
    f = z5py.File(output_container, use_zarr_format=False)
    if not output_dataset in f:
        f.create_dataset(
            output_dataset,
            shape=output_shape,
            compression='gzip',
            dtype='float32',
            chunks=network_output_shape + (num_channels,))
        print("Created data set %s in container %s", output_dataset, output_container)

    # make offset list
    get_offset_lists(output_shape[:-1], gpus, offsets_dir, output_shape=network_output_shape)


    for gpu in gpus:
        offset_file = os.path.join(offsets_dir, 'list_gpu_%i.json' % gpu)
        with open(offset_file, 'r') as f:
            offset_list = json.load(f)

        prediction = TensorflowPredict(inference_graph_basename=meta_graph_basename,
                                       weight_graph_basename=weight_graph_basename,
                                       input_key=input_dataset,
                                       output_key=output_dataset)
        def thread_target():
            t_predict = time.time()
            run_inference_n5(prediction,
                             preprocess,
                             postprocess=None,
                             raw_path=input_container,
                             save_file=output_container,
                             input_key=input_dataset,
                             target_keys=output_dataset,
                             offset_list=offset_list,
                             input_shape=network_input_shape,
                             output_shape=network_output_shape,
                             voxel_size_ratio_input_over_output=voxel_size_ratios)
            t_predict = time.time() - t_predict
            print("Finished inference on %d blocks in %f s on gpu %d" % (len(offset_list), t_predict, gpu))

        t = Thread(target=thread_target(), name='gpu_inference_%d' % gpu)
        t.start()



def gpu_inference_main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('experiment')
    parser.add_argument('input_container')
    parser.add_argument('input_dataset')
    parser.add_argument('output_dataset')
    parser.add_argument('--setup-id', type=int, required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--num-channels', required=True, type=int)
    parser.add_argument('--gpus', type=int, required=True, nargs='+')
    parser.add_argument('--output-container', default=None)
    parser.add_argument('--network-output-shape', type=int, nargs=3, default=(209, 214, 214))
    parser.add_argument('--network-input-shape', type=int, nargs=3, default=(91, 862, 862))
    parser.add_argument('--network-meta-graph-pattern', type=str, default='unet-inference')
    parser.add_argument('--network-weight-graph-pattern', type=str, default='unet_checkpoint_%d')
    parser.add_argument('--offsets-dir', default=None)
    parser.add_argument('--input-voxel-size', default=(360, 36, 36), type=float, nargs=3)
    parser.add_argument('--output-voxel-size', default=(120, 108, 108), type=float, nargs=3)

    args = parser.parse_args()

    input_container  = args.input_container
    input_dataset    = args.input_dataset
    output_container = input_container if args.output_container is None else args.output_container
    output_dataset   = args.output_dataset

    experiment_directory  = os.path.join(args.experiment, str(args.setup_id))
    meta_graph_basename   = os.path.join(experiment_directory, args.network_meta_graph_pattern)
    weight_graph_basename = os.path.join(experiment_directory, args.network_weight_graph_pattern % args.iteration,)
    offsets_dir           = os.path.join(experiment_directory, '.offsets') if args.offsets_dir is None else args.offsets_dir
    print("LOL OFFSETS DIR", offsets_dir)

    gpu_inference(
        meta_graph_basename=meta_graph_basename,
        weight_graph_basename=weight_graph_basename,
        input_container=input_container,
        output_container=output_container,
        input_dataset=input_dataset,
        output_dataset=output_dataset,
        input_voxel_size=args.input_voxel_size,
        output_voxel_size=args.output_voxel_size,
        offsets_dir=offsets_dir,
        gpus=args.gpus,
        network_input_shape=args.network_input_shape,
        network_output_shape=args.network_output_shape,
        num_channels=args.num_channels)

if __name__ == "__main__":
    gpu_inference_main()
