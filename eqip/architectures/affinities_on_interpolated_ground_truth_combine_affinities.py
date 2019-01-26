import logging
_logger = logging.getLogger(__name__)

import networks.ops3d as ops3d
import networks.unet as unet
import tensorflow as tf
import json


class Network(object):
    def __init__(
            self,
            input_shape,
            num_final_features,
            nn_affinity_channels):
        super(Network, self).__init__()

        self.raw = tf.placeholder(tf.float32, shape=input_shape)
        raw_batched = tf.reshape(self.raw, (1, 1,) + input_shape)
        _logger.info("raw shape: %s", self.raw.get_shape().as_list())

        voxel_size = (10, 1, 1)
        fov = (10, 1, 1)
        initial_feature_maps = 12
        fmap_inc_factor = 6

        # first convolve and downsample
        convolved, fov = ops3d.conv_pass(raw_batched, [(1, 3, 3)], initial_feature_maps, activation='relu',
                                         name='conv_1', fov=fov, voxel_size=voxel_size)
        convolved, fov = ops3d.conv_pass(convolved, [(1, 3, 3)], initial_feature_maps, activation='relu', name='conv_2',
                                         fov=fov, voxel_size=voxel_size)
        first_downsampled, fov, anisotropy = ops3d.downsample(convolved, (1, 3, 3), name='initial_down', fov=fov,
                                                              voxel_size=(10, 1, 1))
        anisotropy_remembered = anisotropy
        fov_remembered = fov

        last_fmap, fov, anisotropy = unet.unet(first_downsampled, initial_feature_maps * fmap_inc_factor,
                                               fmap_inc_factor, [[1, 3, 3], [3, 3, 3]],
                                               [[(1, 3, 3), (1, 3, 3)],
                                                [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                               [[(1, 3, 3), (1, 3, 3)],
                                                [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                                               voxel_size=anisotropy, fov=fov)

        _logger.info('last_fmap shape: %s', last_fmap.get_shape().as_list())

        upsampled_fmap, voxel_size = ops3d.upsample(last_fmap, (3, 1, 1), num_fmaps=initial_feature_maps,
                                                    name='final_up', fov=fov_remembered,
                                                    voxel_size=anisotropy_remembered)
        _logger.info("upsampled_fmap shape: %s", upsampled_fmap.get_shape().as_list())
        # TODO for now just use fov_rememered, probably wrong fov though
        convolved_last, fov = ops3d.conv_pass(upsampled_fmap, [(3, 3, 3), (3, 3, 3)], initial_feature_maps,
                                              activation='relu', name='conv_last', fov=fov_remembered,
                                              voxel_size=voxel_size)
        _logger.info("convolved_last shape: %s", convolved_last.get_shape().as_list())

        affinities, fov = ops3d.conv_pass(
            convolved_last,
            kernel_size=[[1, 1, 1]],
            num_fmaps=num_final_features,
            activation=None,
            fov=fov,
            voxel_size=voxel_size
        )

        nearest_neighbor_affinities, fov = ops3d.conv_pass(
            affinities,
            kernel_size=[[3, 3, 3]],
            num_fmaps=3,
            activation=None,
            fov=fov,
            voxel_size=voxel_size,
            name='conv_for_nn')

        cropped_shape = affinities.get_shape().as_list()
        cropped_shape[-3] = cropped_shape[-3] - 2
        self.affinities_cropped = ops3d.crop_zyx(affinities, cropped_shape)

        output_shape_batched = affinities.get_shape().as_list()
        output_shape = output_shape_batched[1:]  # strip the batch dimension
        self.affinities_no_batch = tf.reshape(affinities, output_shape)

        output_shape_nn_batched = nearest_neighbor_affinities.get_shape().as_list()
        output_shape_nn = output_shape_nn_batched[1:]
        self.affinities_nn_no_batch = tf.reshape(nearest_neighbor_affinities, output_shape_nn)
        _logger.info("affinities shape:          %s", output_shape)
        _logger.info('affinities cropped shape:  %s', cropped_shape[1:])
        _logger.info("nn affinities shape:       %s", output_shape_nn)

        self.gt_affinities = tf.placeholder(tf.float32, shape=output_shape)
        # affinities_mask is used for both mask and scale (for balanced loss)
        self.affinities_mask = tf.placeholder(tf.float32, shape=output_shape)

        self.gt_affinities_nn = self._slice_and_concat(self.gt_affinities[..., 1:-1, 1:-1, 1:-1], nn_affinity_channels, 0)
        self.affinities_mask_nn = self._slice_and_concat(self.affinities_mask[..., 1:-1, 1:-1, 1:-1], nn_affinity_channels, 0)
        # TODO update tensorflow on docker image to use tf.uint64
        self.gt_labels = tf.placeholder(tf.int64, shape=output_shape[1:])

        self.mse_loss, self.mse_affs, self.mse_affs_nn, self.mse_optimizer = self._add_mse_loss(
            affinities=self.affinities_no_batch,
            affinities_nn=self.affinities_nn_no_batch,
            gt_affinities=self.gt_affinities,
            gt_affinities_nn=self.gt_affinities_nn,
            affinities_mask=self.affinities_mask,
            affinities_mask_nn=self.affinities_mask_nn,
            optimizer_name='mse_loss')

        tf.summary.scalar('summary_mse_loss', self.mse_loss)
        self.merged = tf.summary.merge_all()

    def _slice_and_concat(self, tensor, indices, dim):
        slices = [self._get_slice(tensor, idx, dim) for idx in indices]
        _logger.debug('Making stack out of slices %s', slices)
        stacked = tf.concat(slices, axis=dim)
        _logger.debug('Stacked %s into %s along dim %d', tensor, stacked, dim)
        return stacked

    def _get_slice(self, tensor, index, dim):
        slicing = [slice(None),] * len(tensor.get_shape().as_list())
        slicing[dim] = slice(index, index+1)
        _logger.debug('Slicing is %s', slicing)
        sliced = tensor[slicing]
        _logger.debug('sliced %s into %s along dim %d', tensor, sliced, dim)
        return sliced



    def export_meta_graph(self, meta_graph_filename):
        tf.train.export_meta_graph(filename=meta_graph_filename)

    def export_io_names(
            self,
            net_io_names,
            io_key_raw,
            io_key_affinities,
            io_key_gt_affinities,
            io_key_affinities_mask,
            io_key_gt_labels,
            io_key_mse_prefix,
            io_key_optimizer,
            io_key_loss,
            io_key_loss_all,
            io_key_loss_nn,
            io_key_summary,
            io_key_affinities_cropped,
            io_key_affinities_nn,
            io_key_gt_affinities_nn,
            io_key_affinities_mask_nn):
        names = {
            io_key_raw: self.raw.name,
            io_key_affinities: self.affinities_no_batch.name,
            io_key_gt_affinities: self.gt_affinities.name,
            io_key_affinities_mask: self.affinities_mask.name,
            io_key_gt_labels: self.gt_labels.name,
            '%s_%s' % (io_key_mse_prefix, io_key_optimizer):self. mse_optimizer.name,
            '%s_%s' % (io_key_mse_prefix, io_key_loss): self.mse_loss.name,
            '%s_%s' % (io_key_mse_prefix, io_key_loss_all): self.mse_affs.name,
            '%s_%s' % (io_key_mse_prefix, io_key_loss_nn): self.mse_affs_nn.name,
            io_key_summary: self.merged.name,
            io_key_affinities_cropped: self.affinities_cropped.name,
            io_key_affinities_nn: self.affinities_nn_no_batch.name,
            io_key_gt_affinities_nn: self.gt_affinities_nn.name,
            io_key_affinities_mask_nn: self.affinities_mask_nn.name
        }

        with open(net_io_names, 'w') as f:
            json.dump(names, f)

    def _add_mse_loss(
            self,
            affinities,
            gt_affinities,
            affinities_nn,
            gt_affinities_nn,
            affinities_mask,
            affinities_mask_nn,
            optimizer_name):
        loss_balanced_all = tf.losses.mean_squared_error(
            labels=gt_affinities,
            predictions=affinities,
            weights=affinities_mask)

        _logger.info('nearest neighbor affinity shape       %s', affinities_nn)
        _logger.info('nearest neighbor affinity gt shape    %s', gt_affinities_nn)
        _logger.info('nearest neighbor affinity mask shape  %s', affinities_mask_nn)

        loss_balanced_nn = tf.losses.mean_squared_error(
            labels=gt_affinities_nn,
            predictions=affinities_nn,
            weights=affinities_mask_nn)

        combined_loss = loss_balanced_all * loss_balanced_nn

        opt = tf.train.AdamOptimizer(
            learning_rate=0.5e-4,
            beta1=0.95,
            beta2=0.999,
            epsilon=1e-8,
            name='%s_adam_optimizer' % optimizer_name)

        optimizer = opt.minimize(combined_loss)

        return loss_balanced_all, loss_balanced_nn, combined_loss, optimizer

def _mk_net(
        meta_graph_filename,
        net_io_names,
        io_key_mse_prefix,
        io_key_raw,
        io_key_affinities,
        io_key_affinities_nn,
        io_key_gt_affinities,
        io_key_gt_affinities_nn,
        io_key_affinities_mask,
        io_key_affinities_mask_nn,
        io_key_optimizer,
        io_key_loss,
        io_key_loss_all,
        io_key_loss_nn,
        io_key_summary,
        io_key_gt_labels,
        io_key_affinities_cropped,
        num_final_features,
        nn_affinity_channels):
    input_shape = (43, 430, 430)
    network = Network(
        input_shape=input_shape,
        num_final_features=num_final_features,
        nn_affinity_channels=nn_affinity_channels)
    network.export_meta_graph(meta_graph_filename)

    network.export_io_names(
        net_io_names=net_io_names,
        io_key_raw=io_key_raw,
        io_key_affinities=io_key_affinities,
        io_key_gt_affinities=io_key_gt_affinities,
        io_key_affinities_mask=io_key_affinities_mask,
        io_key_gt_labels=io_key_gt_labels,
        io_key_mse_prefix=io_key_mse_prefix,
        io_key_optimizer=io_key_optimizer,
        io_key_loss=io_key_loss,
        io_key_loss_all=io_key_loss_all,
        io_key_loss_nn=io_key_loss_nn,
        io_key_summary=io_key_summary,
        io_key_affinities_cropped=io_key_affinities_cropped,
        io_key_affinities_nn=io_key_affinities_nn,
        io_key_gt_affinities_nn=io_key_gt_affinities_nn,
        io_key_affinities_mask_nn=io_key_affinities_mask_nn
    )

def _inference_net(unet_inference_meta, num_final_features, nn_affinity_channels):
    input_shape = (91, 862, 862)
    network = Network(
        input_shape=input_shape,
        num_final_features=num_final_features,
        nn_affinity_channels=nn_affinity_channels)
    network.export_meta_graph(unet_inference_meta)


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
    parser.add_argument('--io-key-affinities-nn', type=str, default='affinities_nn')
    parser.add_argument('--io-key-gt-affinities', type=str, default='gt_affinities')
    parser.add_argument('--io-key-affinities-mask', type=str, default='affinities_mask')
    parser.add_argument('--io-key-gt-affinities-nn', type=str, default='gt_affinities_nn')
    parser.add_argument('--io-key-affinities-mask-nn', type=str, default='affinities_mask_nn')

    parser.add_argument('--io-key-affinities-cropped', type=str, default='affinities_cropped')
    parser.add_argument('--io-key-optimizer', type=str, default='optimizer')
    parser.add_argument('--io-key-loss', type=str, default='loss')
    parser.add_argument('--io-key-loss-all', type=str, default='loss-all')
    parser.add_argument('--io-key-loss-nn', type=str, default='loss-nn')
    parser.add_argument('--io-key-summary', type=str, default='summary')
    parser.add_argument('--io-key-gt-labels', type=str, default='gt_labels')
    parser.add_argument('--num-affinities', type=int, default=3)

    args = parser.parse_args()

    # for now just hard code slicing
    nn_affinity_channels = tuple(range(0, args.num_affinities, args.num_affinities // 3))

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
        io_key_affinities_cropped = args.io_key_affinities_cropped,
        nn_affinity_channels      = nn_affinity_channels,
        io_key_loss_all           = args.io_key_loss_all,
        io_key_loss_nn            = args.io_key_loss_nn,
        io_key_affinities_nn      = args.io_key_affinities_nn,
        io_key_gt_affinities_nn   = args.io_key_gt_affinities_nn,
        io_key_affinities_mask_nn = args.io_key_affinities_mask_nn
    )
    tf.reset_default_graph()

    if args.inference_meta_graph_filename is not None:
        _inference_net(
            num_final_features   = args.num_affinities,
            unet_inference_meta  = args.inference_meta_graph_filename,
            nn_affinity_channels = nn_affinity_channels)

    _logger.info('Using tensorflow version %s', tf.__version__)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    make()
