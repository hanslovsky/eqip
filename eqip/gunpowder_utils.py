import logging
_logger = logging.getLogger(__name__)

import glob
import itertools

from gunpowder import ArrayKey, ArraySpec, Hdf5Source

from fuse import Z5Source



RAW_KEY                = ArrayKey('RAW')
DEFECT_MASK_KEY        = ArrayKey('DEFECT_MASK')
LABELS_KEY             = ArrayKey('LABELS')
NEURON_IDS_NO_GLIA_KEY = ArrayKey('NEURON_IDS_NOGLIA')
MASK_KEY               = ArrayKey('MASK')
TRAINING_MASK_KEY      = ArrayKey('TRAINING_MASK')
LOSS_GRADIENT_KEY      = ArrayKey('LOSS_GRADIENT')
AFFINITIES_KEY         = ArrayKey('AFFINITIES')
GT_AFFINITIES_KEY      = ArrayKey('GT_AFFINITIES')
AFFINITIES_MASK_KEY    = ArrayKey('AFFINITIES_MASK')
AFFINITIES_SCALE_KEY   = ArrayKey('AFFINITIES_SCALE')
AFFINITIES_NN_KEY      = ArrayKey('AFFINITIES_NN')
GLIA_MASK_KEY          = ArrayKey('GLIA_MASK')
GLIA_KEY               = ArrayKey('GLIA')
GLIA_SCALE_KEY         = ArrayKey('GLIA_SCALE')
GT_GLIA_KEY            = ArrayKey('GT_GLIA')

_LOCALS = dict(locals())
_lower_key_identifier_to_key = {v.identifier.lower() : v for _, v in _LOCALS.items() if isinstance(v, ArrayKey)}
_logger.info('Lower key identifier to key mapping: %s', _lower_key_identifier_to_key)


DEFAULT_PATHS = {
    RAW_KEY.identifier.lower()                : 'volumes/raw',
    LABELS_KEY.identifier.lower()             : 'volumes/labels/neuron_ids-downsampled',
    MASK_KEY.identifier.lower()               : 'volumes/labels/mask-downsampled',
    GT_GLIA_KEY.identifier.lower()            : 'volumes/labels/glia-downsampled',
    GLIA_MASK_KEY.identifier.lower()          : 'volumes/labels/mask-downsampled',
    NEURON_IDS_NO_GLIA_KEY.identifier.lower() : 'volumes/labels/neuron_ids_noglia-downsampled'
}

def make_data_providers(
        *provider_strings,
        required_paths=('raw', 'labels', 'mask'),
        default_paths=DEFAULT_PATHS,
        lower_key_identifier_to_key=_lower_key_identifier_to_key):
    return tuple(itertools.chain.from_iterable(tuple(make_data_provider(s, required_paths, default_paths, lower_key_identifier_to_key) for s in provider_strings)))


def make_data_provider(provider_string, required_paths, default_paths, lower_key_identifier_to_key):
    data_providers = []
    # data_dir = '/groups/saalfeld/home/hanslovskyp/experiments/quasi-isotropic/data/realigned'
    # file_pattern = '*merged*fixed-offset-fixed-mask.h5'


    pattern = provider_string.split(':')[0]
    paths   = {r: default_paths[r] for r in required_paths if r in default_paths}
    paths.update(**{entry.split('=')[0].lower() : entry.split('=')[1] for entry in provider_string.split(':')[1:]})

    for r in required_paths:
        assert r in paths, "Required path {} not provided in {}".format(r, paths)

    for k, _ in paths.items():
        assert k in lower_key_identifier_to_key, "Identifier %s not in array key map %s" % (k, lower_key_identifier_to_key)

    datasets = {lower_key_identifier_to_key[k] : v for k, v in paths.items()}

    specs = {
        MASK_KEY:      ArraySpec(interpolatable=False),
        GLIA_MASK_KEY: ArraySpec(interpolatable=False)
    }

    for data in glob.glob(pattern):
        if data.endswith('h5') or data.endswith('hdf'):
            source = Hdf5Source(data, datasets=datasets, array_specs=specs)
        else:
            source = Z5Source(data, datasets=datasets, array_specs=specs, revert=False)
        data_providers.append(source)

    return tuple(data_providers)