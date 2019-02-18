import logging
_logger = logging.getLogger(__name__)

import subprocess

from .version_info import _version as version
# git clone https://github.com/saalfeldlab/CNNectome
# somehow cython does not depend on conda's gcc, will have to explicitly
# add gcc as dependency (gxx_linux-64):
# https://github.com/Anaconda-Platform/anaconda-project/issues/183#issuecomment-462796564
# if there is a better dependency than gxx_linux-64 (to make it independent of system), please let me know!
_job_template=r'''#!/usr/bin/env bash

set -e

if [ -f "%(conda_sh)s" ]; then
    . "%(conda_sh)s"
fi

conda create \
      --%(name_or_prefix)s=%(name)s \
      -c conda-forge \
      -y \
      python=3.6
conda activate %(name)s
pip install h5py scikit-image numpy scipy requests urllib3
conda install -c conda-forge -y z5py gxx_linux-64 cython tensorflow-gpu=1.3
pip install git+https://github.com/hanslovsky/eqip@%(eqip_revision)s
'''

_clone_environment_job_template=r'''#!/usr/bin/env bash

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

default_revisions = {'eqip' : version.version() if version.tag() == '' else 'master'}

def create_eqip_environment(
        name,
        use_name_as_prefix=False,
        conda_sh = '$HOME/miniconda3/etc/profile.d/conda.sh',
        eqip_revision=default_revisions['eqip']):

    script = _job_template % dict(
        conda_sh       = conda_sh,
        name_or_prefix = 'prefix' if use_name_as_prefix else 'name',
        name           = name,
        eqip_revision  = eqip_revision)

    _logger.debug('conda env create script: %s', script)
    # TODO find a way to do this without shell=True
    p = subprocess.Popen(script, shell=True, executable='/bin/bash')
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
    # TODO find a way to do this without shell=True
    p = subprocess.Popen(script, shell=True, executable='/bin/bash')
    stdout, stderr = p.communicate()
    # _logger.debug('stdout: %s', stdout)
    # _logger.debug('stderr: %s', stderr)




if __name__ == "__main__":
    logging.basicConfig()
    _logger.setLevel(logging.DEBUG)
    create_eqip_environment(name='conda-env-eqip-test', use_name_as_prefix=False)
    clone_eqip_environment(name='conda-env-eqip-test-clone', use_name_as_prefix=False, clone_from='conda-env-eqip-test', extra_pip_installs=('pandas',))
