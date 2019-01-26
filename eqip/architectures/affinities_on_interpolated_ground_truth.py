import networks.ops3d as ops3d
import networks.unet as unet
import tensorflow as tf
import json

def _mk_net(
        meta_graph_filename,
        net_io_names,
        io_key_mse_prefix,
        io_key_raw,
        io_key_affinities,
        io_key_gt_affinities,
        io_key_affinities_mask,
        io_key_optimizer,
        io_key_loss,
        io_key_summary,
        io_key_gt_labels,
        io_key_affinities_cropped,
        num_final_features
           ):
    input_shape = (43, 430, 430)
    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_batched = tf.reshape(raw, (1, 1,) + input_shape)
    print("raw shape: ", raw.get_shape().as_list())


    def add_mse_loss(affinities, gt_affinities, affinities_mask):
        loss_balanced = tf.losses.mean_squared_error(
            labels = gt_affinities,
            predictions = affinities,
            weights = affinities_mask
        )

        opt = tf.train.AdamOptimizer(
            learning_rate = 0.5e-4,
            beta1         = 0.95,
            beta2         = 0.999,
            epsilon       = 1e-8,
            name          = '%s_adam_optimizer' % io_key_mse_prefix)

        optimizer = opt.minimize(loss_balanced)

        return loss_balanced, optimizer

    voxel_size = (10, 1, 1)
    fov = (10, 1, 1)
    initial_feature_maps = 12
    fmap_inc_factor = 6

    # first convolve and downsample
    convolved, fov = ops3d.conv_pass(raw_batched, [(1, 3, 3)], initial_feature_maps, activation='relu', name='conv_1', fov=fov, voxel_size=voxel_size)
    convolved, fov = ops3d.conv_pass(convolved, [(1, 3, 3)], initial_feature_maps, activation='relu', name='conv_2', fov=fov, voxel_size=voxel_size)
    first_downsampled, fov, anisotropy = ops3d.downsample(convolved, (1, 3, 3), name='initial_down', fov=fov, voxel_size=(10, 1, 1))
    anisotropy_remembered = anisotropy
    fov_remembered = fov

    last_fmap, fov, anisotropy = unet.unet(first_downsampled, initial_feature_maps * fmap_inc_factor, fmap_inc_factor, [[1, 3, 3], [3, 3, 3]],
                                           [[(1, 3, 3), (1, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           [[(1, 3, 3), (1, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           voxel_size=anisotropy, fov=fov)

    print('last_fmap shape:', last_fmap.get_shape().as_list())

    upsampled_fmap, voxel_size = ops3d.upsample(last_fmap, (3, 1, 1), num_fmaps=initial_feature_maps, name='final_up', fov=fov_remembered, voxel_size=anisotropy_remembered)
    print("upsampled_fmap shape:", upsampled_fmap.get_shape().as_list())
    # TODO for now just use fov_rememered, probably wrong fov though
    convolved_last, fov = ops3d.conv_pass(upsampled_fmap, [(3, 3, 3), (3, 3, 3)], initial_feature_maps, activation='relu', name='conv_last', fov=fov_remembered, voxel_size=voxel_size)
    print("convolved_last shape:", convolved_last.get_shape().as_list())

    affinities, fov = ops3d.conv_pass(
            convolved_last,
            kernel_size=[[1, 1, 1]],
            num_fmaps=num_final_features,
            activation=None,
            fov=fov,
            voxel_size=voxel_size
            )

    cropped_shape = affinities.get_shape().as_list()
    cropped_shape[-3] = cropped_shape[-3] - 2
    affinities_cropped = ops3d.crop_zyx(affinities, cropped_shape)

    output_shape_batched = affinities.get_shape().as_list()
    output_shape = output_shape_batched[1:]  # strip the batch dimension
    affinities_no_batch = tf.reshape(affinities, output_shape)
    print("affinities shape:", output_shape)

    gt_affinities     = tf.placeholder(tf.float32, shape=output_shape)
    # affinities_mask is used for both mask and scale (for balanced loss)
    affinities_mask   = tf.placeholder(tf.float32, shape=output_shape)
    # TODO update tensorflow on docker image to use tf.uint64
    gt_labels       = tf.placeholder(tf.int64, shape=output_shape[1:])

    mse_loss, mse_optimizer = add_mse_loss(affinities=affinities_no_batch, gt_affinities=gt_affinities, affinities_mask=affinities_mask)

    tf.summary.scalar('summary_%s_%s' % (io_key_mse_prefix, io_key_loss), mse_loss)
    merged = tf.summary.merge_all()

    tf.train.export_meta_graph(filename=meta_graph_filename)

    names = {
        io_key_raw                                        : raw.name,
        io_key_affinities                                 : affinities_no_batch.name,
        io_key_gt_affinities                              : gt_affinities.name,
        io_key_affinities_mask                            : affinities_mask.name,
        io_key_gt_labels                                  : gt_labels.name,
        '%s_%s' % (io_key_mse_prefix, io_key_optimizer)   : mse_optimizer.name,
        '%s_%s' % (io_key_mse_prefix, io_key_loss)        : mse_loss.name,
        io_key_summary                                    : merged.name,
        io_key_affinities_cropped                         : affinities_cropped.name
    }

    with open(net_io_names, 'w') as f:
        json.dump(names, f)


def _inference_net(unet_inference_meta, num_final_features):
    input_shape = (91, 862, 862)
    raw = tf.placeholder(tf.float32, shape=input_shape)
    raw_batched = tf.reshape(raw, (1, 1,) + input_shape)

    voxel_size = (10, 1, 1)
    fov = (10, 1, 1)
    initial_feature_maps = 12
    fmap_inc_factor = 6

    # first convolve and downsample
    convolved, fov = ops3d.conv_pass(raw_batched, [(1, 3, 3)], initial_feature_maps, activation='relu', name='conv_1', fov=fov, voxel_size=voxel_size)
    convolved, fov = ops3d.conv_pass(convolved, [(1, 3, 3)], initial_feature_maps, activation='relu', name='conv_2', fov=fov, voxel_size=voxel_size)
    first_downsampled, fov, anisotropy = ops3d.downsample(convolved, (1, 3, 3), name='initial_down', fov=fov, voxel_size=(10, 1, 1))
    anisotropy_remembered = anisotropy
    fov_remembered = fov

    last_fmap, fov, anisotropy = unet.unet(first_downsampled, initial_feature_maps * fmap_inc_factor, fmap_inc_factor, [[1, 3, 3], [3, 3, 3]],
                                           [[(1, 3, 3), (1, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           [[(1, 3, 3), (1, 3, 3)],
                                            [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                           voxel_size=anisotropy, fov=fov)

    upsampled_fmap, voxel_size = ops3d.upsample(last_fmap, (3, 1, 1), num_fmaps=initial_feature_maps, name='final_up',
                                                fov=fov_remembered, voxel_size=anisotropy_remembered)
    print("upsampled_fmap shape:", upsampled_fmap.get_shape().as_list())
    # TODO for now just use fov_remembered, probably wrong fov though
    convolved_last, fov = ops3d.conv_pass(upsampled_fmap, [(3, 3, 3), (3, 3, 3)], initial_feature_maps,
                                          activation='relu', name='conv_last', fov=fov_remembered,
                                          voxel_size=voxel_size)
    print("convolved_last shape:", convolved_last.get_shape().as_list())

    affinities, fov = ops3d.conv_pass(
        convolved_last,
        kernel_size=[[1, 1, 1]],
        num_fmaps=num_final_features,
        activation=None,
        fov=fov,
        voxel_size=voxel_size
    )

    cropped_shape = affinities.get_shape().as_list()
    cropped_shape[-3] = cropped_shape[-3] - 2
    affinities = ops3d.crop_zyx(affinities, cropped_shape)

    output_shape_batched = affinities.get_shape().as_list()
    output_shape = output_shape_batched[1:]  # strip the batch dimension
    print("affinities shape:", output_shape)
    # TODO do I need this reshape? For loading weights from checkpoints?
    affinities_no_batch = tf.reshape(affinities, output_shape)

    # 'unet_inference.meta'
    tf.train.export_meta_graph(filename=unet_inference_meta)


def make():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-graph-filename', default='unet.meta', type=str, help='Filename with information about meta graph for network.')
    parser.add_argument('--inference-meta-graph-filename', default='unet-inference.meta', type=str, metavar='FILENAME')
    parser.add_argument('--optimizer-name', type=str, help='name parameter of the tensorflow adam optimizer.', default=None)
    parser.add_argument('--log-level', choices=('DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'), default='INFO', type=str)
    parser.add_argument('--net-io-names', type=str, default='net_io_names.json', help='Path to file holding network input/output name specs')
    parser.add_argument('--io-key-mse-prefix', type=str, default='mse')
    parser.add_argument('--io-key-raw', type=str, default='raw')
    parser.add_argument('--io-key-affinities', type=str, default='affinities')
    parser.add_argument('--io-key-gt-affinities', type=str, default='gt_affinities')
    parser.add_argument('--io-key-affinities-mask', type=str, default='affinities_mask')

    parser.add_argument('--io-key-affinities-cropped', type=str, default='affinities_cropped')
    parser.add_argument('--io-key-optimizer', type=str, default='optimizer')
    parser.add_argument('--io-key-loss', type=str, default='loss')
    parser.add_argument('--io-key-summary', type=str, default='summary')
    parser.add_argument('--io-key-gt-labels', type=str, default='gt_labels')
    parser.add_argument('--num-affinities', type=int, default=3)

    args = parser.parse_args()

    _mk_net(
        num_final_features        = args.num_affinities,
        meta_graph_filename       = args.meta_graph_filename,
        net_io_names              = args.net_io_names,
        io_key_mse_prefix         = args.io_key_mse_prefix,
        io_key_raw                = args.io_key_raw,
        io_key_affinities         = args.io_key_affinities,
        io_key_gt_affinities      = args.io_key_gt_affinities,
        io_key_affinities_mask    = args.io_key_affinities_mask,
        io_key_gt_labels          = args.io_key_gt_labels,
        io_key_loss               = args.io_key_loss,
        io_key_optimizer          = args.io_key_optimizer,
        io_key_summary            = args.io_key_summary,
        io_key_affinities_cropped = args.io_key_affinities_cropped
    )
    tf.reset_default_graph()

    if args.inference_meta_graph_filename is not None:
        _inference_net(
            num_final_features  = args.num_affinities,
            unet_inference_meta = args.inference_meta_graph_filename)

    print('Using tensorflow version', tf.__version__)

if __name__ == '__main__':
    make()
