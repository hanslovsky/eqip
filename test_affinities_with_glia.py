import os

import eqip.architectures
import eqip.training

import logging

# parser.add_argument('--meta-graph-filename', default='unet.meta', type=str,
#                     help='Filename with information about meta graph for network.')
# parser.add_argument('--inference-meta-graph-filename', default='unet-inference.meta', type=str, metavar='FILENAME')
# parser.add_argument('--optimizer-name', type=str, help='name parameter of the tensorflow adam optimizer.', default=None)
# parser.add_argument('--log-level', choices=('DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'), default='INFO', type=str)
# parser.add_argument('--net-io-names', type=str, default='net_io_names.json',
#                     help='Path to file holding network input/output name specs')
# parser.add_argument('--num-affinities', type=int, default=3)

experiment_dir = os.path.join(os.getcwd(), 'test-affinities-with-glia')
print("Experiment dir:", experiment_dir)
unet_meta = os.path.join(experiment_dir, 'unet.meta')
unet_inference_meta = os.path.join(experiment_dir, 'unet-inference.meta')
net_io_names = os.path.join(experiment_dir, 'net_io_names.json')

mknet_argv = (
    # '--help',
    '--meta-graph-filename=%s' % unet_meta,
    '--inference-meta-graph-filename=%s' % unet_inference_meta,
    '--net-io-names=%s' % net_io_names,
    '--num-affinities=3')

eqip.architectures.affinities_on_interpolated_ground_truth_with_glia(argv=mknet_argv)

unet_meta = os.path.join(experiment_dir, 'unet')
train_net_argv = (
    # '--help',
    '--data-provider=/groups/saalfeld/home/hanslovskyp/data/from-arlo/interpolated-combined/sample_*.n5:MASK=volumes/labels/mask-downsampled-75%-y:GLIA_MASK=volumes/labels/mask-downsampled-75%-y',
    '--log-level=DEBUG',
    '--training-directory=%s' % experiment_dir,
    '--meta-graph-filename=%s' % unet_meta,
    '--mse-iterations=200000',
    '--malis-iterations=0',
    '--net-io-names=%s' % net_io_names,
    '--save-checkpoint-every=200',
    '--pre-cache-num-workers=1',
    '--pre-cache-size=1',
    '--snapshot-every=10',
    '--ignore-labels-for-slip',
    '--grow-boundaries=0',
    '--mask-out-labels', '0')

import logging
logging.basicConfig()
# logging.getLogger('gunpowder.nodes').setLevel(logging.DEBUG)

eqip.training.affinities_on_interpolated_ground_truth_with_glia(argv=train_net_argv)