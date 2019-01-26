import os
from setuptools import setup


install_requires = [
    'gunpowder',
    'scikit-image>=0.14.1',
    'numpy>=1.15.4',
    'scipy>=1.1.0',
    'daisy',
    'malis',
    'augment @ git+https://github.com/funkey/augment'
    #'z5py',
]

console_scripts = [
    'make-affinities-on-interpolated-ground-truth=eqip.architectures:affinities_on_interpolated_ground_truth',
    'make-affinities-on-interpolated-ground-truth-combine-affinities=eqip.architectures:affinities_on_interpolated_ground_truth_combine_affinities',
    'train-affinities-on-interpolated-ground-truth=eqip.training:affinities_on_interpolated_ground_truth',
    'train-affinities-on-interpolated-ground-truth-combine-affinities=eqip.training:affinities_on_interpolated_ground_truth_combine_affinities',
    'create-setup=eqip:create_setup',
    'predict-affinities=eqip.inference:predict_affinities_daisy',
    'list-latest-checkpoint=eqip:list_latest_checkpoint',
    'list-latest-snapshot=eqip:list_latest_snapshot'
]

entry_points = dict(console_scripts=console_scripts)

packages = [
    'eqip',
    'eqip.architectures',
    'eqip.training',
    'eqip.inference'
]

here = os.path.abspath(os.path.dirname(__file__))
version = {}
with open(os.path.join(here, 'eqip', 'version.py')) as fp:
    exec(fp.read(), version)

setup(
    name='eqip',
    version=version['__version__'],
    author='Philipp Hanslovsky',
    author_email='hanslovskyp@janelia.hhmi.org',
    description='',
    url='https://github.com/hanslovsky/eqip',
    install_requires=install_requires,
    entry_points=entry_points,
    packages=packages
)
