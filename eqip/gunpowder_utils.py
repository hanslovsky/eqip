import glob
import itertools

from gunpowder import ArrayKey, ArraySpec, Hdf5Source



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
AFFINITIES_NN_KEY    = ArrayKey('AFFINITIES_NN')

DEFAULT_PATHS = dict(
    raw    = 'volumes/raw',
    labels = 'volumes/labels/neuron_ids-downsampled',
    mask   = 'volumes/masks/neuron_ids-downsampled')

def make_data_providers(*provider_strings):
    return tuple(itertools.chain.from_iterable(tuple(make_data_provider(s) for s in provider_strings)))


def make_data_provider(provider_string):
    data_providers = []
    # data_dir = '/groups/saalfeld/home/hanslovskyp/experiments/quasi-isotropic/data/realigned'
    # file_pattern = '*merged*fixed-offset-fixed-mask.h5'


    pattern = provider_string.split(':')[0]
    paths   = {**DEFAULT_PATHS}
    paths.update(**{entry.split('=')[0].lower() : entry.split('=')[1] for entry in provider_string.split(':')[1:]})


    for data in glob.glob(pattern):
        h5_source = Hdf5Source(
            data,
            datasets={
                RAW_KEY: paths['raw'],
                GT_LABELS_KEY: paths['labels'],
                GT_MASK_KEY: paths['mask']
                },
            array_specs={
                GT_MASK_KEY: ArraySpec(interpolatable=False)
            }
        )
        data_providers.append(h5_source)
    return tuple(data_providers)