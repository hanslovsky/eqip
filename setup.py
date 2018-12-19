from setuptools import setup
from os import path

install_requires = [
    'gunpowder',
    'scikit-image>=0.14.1',
    'numpy>=1.15.4',
    'scipy>=1.1.0']

setup(
    name='eqip',
    version='0.1.0dev',
    author='Philipp Hanslovsky',
    author_email='hanslovskyp@janelia.hhmi.org',
    description='',
    url='https://github.com/hanslovsky/eqip',
    install_requires=install_requires
)
