import malis
import tensorflow as tf

from . import io_keys

def add_mse_loss_with_adam(
        predictions,
        labels,
        weights,
        opt):
    loss_balanced = tf.losses.mean_squared_error(
        labels = labels,
        predictions = predictions,
        weights = weights)

    opt       = default_adam_optimizer(opt) if isinstance(opt, str) else opt
    optimizer = opt.minimize(loss_balanced)
    return loss_balanced, optimizer

def default_adam_optimizer(name):
    return tf.train.AdamOptimizer(
        learning_rate=0.5e-4,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8,
        name=name)


def loss_affinities_with_glia(
        net_io_names,
        optimizer_or_name,
        summary_name,
        glia_loss_type='mse',
        tf_loss = tf.losses.mean_squared_error):

    def loss_func(graph):
        affinities = graph.get_tensor_by_name(net_io_names[io_keys.AFFINITIES])
        gt_affinities = graph.get_tensor_by_name(net_io_names[io_keys.GT_AFFINITIES])
        affinities_mask = graph.get_tensor_by_name(net_io_names[io_keys.AFFINITIES_MASK])
        loss_affinities = tf_loss(gt_affinities, affinities, affinities_mask)

        glia_loss = graph.get_tensor_by_name(net_io_names[io_keys.glia_loss_name(glia_loss_type)])

        loss = glia_loss * loss_affinities
        opt = default_adam_optimizer(optimizer_or_name) if isinstance(optimizer_or_name, str) else optimizer_or_name
        optimizer = opt.minimize(loss)

        tf.summary.scalar(summary_name, loss)

        return loss, optimizer

    return loss_func

def malis_loss_with_glia(
        net_io_names,
        neighborhood,
        optimizer_or_name,
        summary_name):
    def loss_func(graph):

        gt_labels = graph.get_tensor_by_name(net_io_names[io_keys.GT_LABELS])

        tf_loss = lambda labels, pred, weights: malis.malis_loss_op(
            affs = pred,
            gt_affs = labels,
            gt_seg = gt_labels,
            neighborhood = neighborhood,
            gt_aff_mask = weights)

        functor = loss_affinities_with_glia(net_io_names, optimizer_or_name, summary_name, tf_loss=tf_loss)
        return functor(graph)

    return loss_func
