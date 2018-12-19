#!/usr/bin/env python

import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--repository', default='hanslovsky/eqip')
parser.add_argument('--version', default=None)
parser.add_argument('--revision', default=None)

args = parser.parse_args()

here = os.path.abspath(os.path.dirname(__file__))
os.chdir(here)
version = {}
with open(os.path.join(here, 'eqip', 'version.py')) as fp:
    exec(fp.read(), version)
    
version  = version['__version__'] if args.version is None else args.version
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii') if args.revision is None else args.revision

docker_cmd = [
              'docker',
              'build',
              '--build-arg', 'EQIP_REVISION=%s' % revision,
              '-t', '%s:%s' % (args.repository, version), 
              os.path.join(here, 'docker-container')]

print(docker_cmd)

subprocess.call(docker_cmd)





