import logging
_logger = logging.getLogger(__name__)

import subprocess

from .version import __tag__, __version__

# git clone https://github.com/saalfeldlab/CNNectome
# somehow cython does not depend on conda's gcc, will have to explicitly
# add gcc as dependency (gxx_linux-64):
# https://github.com/Anaconda-Platform/anaconda-project/issues/183#issuecomment-462796564
# if there is a better dependency than gxx_linux-64 (to make it independent of system), please let me know!
_job_template=r'''#!/usr/bin/env sh

set -e

if [ -f "%(conda_sh)s" ]; then
    . "%(conda_sh)s"
fi

conda create \
      --%(name_or_prefix)s=%(name)s \
      -c conda-forge \
      -y \
      python=3.6 \
      h5py \
      z5py \
      scikit-image \
      numpy \
      scipy \
      requests \
      urllib3 \
      gxx_linux-64 \
      cython \
      pip \
      tensorflow-gpu=1.3
conda activate %(name)s
pip install malis==1.0
pip install git+https://github.com/funkey/augment@%(augment_revision)s
pip install git+https://github.com/funkey/gunpowder@%(gunpowder_revision)s
pip install git+https://github.com/hanslovsky/gunpowder-nodes@%(gunpowder_nodes_revision)s
pip install git+https://github.com/funkelab/daisy@%(daisy_revision)s
pip install git+https://github.com/hanslovsky/eqip@%(eqip_revision)s
'''

_clone_environment_job_template=r'''#!/usr/bin/env sh

set -e

if [ -f "%(conda_sh)s" ]; then
    . "%(conda_sh)s"
fi

conda create \
      --%(name_or_prefix)s=%(name)s \
      --clone=%(clone_from)s \
      -c conda-forge \
      -y
conda activate %(name)s
'''

default_revisions = {
        'augment'        : '4a42b01ccad7607b47a1096e904220729dbcb80a',
        'gunpowder'      : 'd49573f53e8f23d12461ed8de831d0103acb2715',
        'gunpowder-nodes': '2d94463ae5a4fbcc0063aaeeb8210a516e0b65aa',
        'daisy'          : '41130e58582ae05d01d26261786de0cbafaa6482',
        'eqip'           : __version__ if __tag__ == '' else 'master'}

def create_eqip_environment(
        name,
        use_name_as_prefix=False,
        conda_sh = '$HOME/miniconda3/etc/profile.d/conda.sh',
        augment_revision=default_revisions['augment'],
        gunpowder_revision=default_revisions['gunpowder'],
        gunpowder_nodes_revision=default_revisions['gunpowder-nodes'],
        eqip_revision=default_revisions['eqip'],
        daisy_revision=default_revisions['daisy']):

    script = _job_template % dict(
        conda_sh                 = conda_sh,
        name_or_prefix           = 'prefix' if use_name_as_prefix else 'name',
        name                     = name,
        augment_revision         = augment_revision,
        gunpowder_revision       = gunpowder_revision,
        gunpowder_nodes_revision = gunpowder_nodes_revision,
        eqip_revision            = eqip_revision,
        daisy_revision           = daisy_revision)

    _logger.debug('conda env create script: %s', script)
    p = subprocess.Popen(script, shell=True)
    stdout, stderr = p.communicate()
    # _logger.debug('stdout: %s', stdout)
    # _logger.debug('stderr: %s', stderr)

def clone_eqip_environment(
        name,
        clone_from,
        extra_pip_installs=(),
        use_name_as_prefix=False,
        conda_sh = '$HOME/miniconda3/etc/profile.d/conda.sh'):
    script = _clone_environment_job_template % dict(
        conda_sh = conda_sh,
        clone_from=clone_from,
        name_or_prefix = 'prefix' if use_name_as_prefix else 'name',
        name = name)

    for epi in extra_pip_installs:
        script += '\npip install %s' % epi

    _logger.debug('conda env clone script: %s', script)
    p = subprocess.Popen(script, shell=True)
    stdout, stderr = p.communicate()
    # _logger.debug('stdout: %s', stdout)
    # _logger.debug('stderr: %s', stderr)




if __name__ == "__main__":
    logging.basicConfig()
    _logger.setLevel(logging.DEBUG)
    create_eqip_environment(name='conda-env-eqip-test', use_name_as_prefix=False)
    clone_eqip_environment(name='conda-env-eqip-test-clone', use_name_as_prefix=False, clone_from='conda-env-eqip-test', extra_pip_installs=('pandas',))
