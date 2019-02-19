import os
from setuptools import setup


install_requires = [
    'pip>=18.1',
    'scikit-image>=0.14.1',
    'numpy>=1.15.4',
    'scipy>=1.1.0',
    'malis @ git+https://github.com/TuragaLab/malis@beb4ee965acee89ab00a20a70205f51177003c69',
    'augment @ git+https://github.com/funkey/augment',
    'gunpowder @ git+https://github.com/funkey/gunpowder@d49573f53e8f23d12461ed8de831d0103acb2715',
    'daisy @ git+https://github.com/funkelab/daisy@41130e58582ae05d01d26261786de0cbafaa6482',
    'fuse @ git+https://github.com/hanslovsky/fuse@0.1.0'
]

console_scripts = [
    'make-affinities-on-interpolated-ground-truth=eqip.architectures:affinities_on_interpolated_ground_truth',
    'make-affinities-on-interpolated-ground-truth-with-glia=eqip.architectures:affinities_on_interpolated_ground_truth_with_glia',
    'make-affinities-on-interpolated-ground-truth-combine-affinities=eqip.architectures:affinities_on_interpolated_ground_truth_combine_affinities',
    'train-affinities-on-interpolated-ground-truth=eqip.training:affinities_on_interpolated_ground_truth',
    'train-affinities-on-interpolated-ground-truth-with-glia=eqip.training:affinities_on_interpolated_ground_truth_with_glia',
    'train-affinities-on-interpolated-ground-truth-combine-affinities=eqip.training:affinities_on_interpolated_ground_truth_combine_affinities',
    'create-setup=eqip:create_setup',
    'predict-daisy=eqip.inference:predict_affinities_daisy',
    'list-latest-checkpoint=eqip:list_latest_checkpoint',
    'list-latest-snapshot=eqip:list_latest_snapshot',
    'create-experiment=eqip.experiment:create_experiment'
]

entry_points = dict(console_scripts=console_scripts)

name = 'eqip'

packages = [
    f'{name}',
    f'{name}.architectures',
    f'{name}.training',
    f'{name}.inference',
    f'{name}.experiment'
]

here = os.path.abspath(os.path.dirname(__file__))
version_info = {}
with open(os.path.join(here, name, 'version_info.py')) as fp:
    exec(fp.read(), version_info)
version = version_info['_version']

with open(os.path.join(here, 'README.md')) as f:
    eqip_long_description = f.read()

setup(
    name=name,
    python_requires='>=3.6',
    version=version.version(),
    author='Philipp Hanslovsky',
    author_email='hanslovskyp@janelia.hhmi.org',
    description='EM Quasi-Isotropic Prediction',
    long_description=eqip_long_description,
    long_description_content_type='text/markdown',
    url=f'https://github.com/hanslovsky/{name}',
    install_requires=install_requires,
    entry_points=entry_points,
    packages=packages
)
