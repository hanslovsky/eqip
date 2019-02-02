#!/usr/bin/env python

# TODO clone repo into tmp directory and then build

import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--repository', default='hanslovsky')
parser.add_argument('--name', required=True)
parser.add_argument('--version', default=None)
parser.add_argument('--revision', default=None)
parser.add_argument('--python', choices=('3.5'), required=True, type=str)
parser.add_argument('--num-make-jobs', required=False, type=int, default=os.cpu_count())

args = parser.parse_args()

here = os.path.abspath(os.path.dirname(__file__))
os.chdir(here)
version = {}
with open(os.path.join(here, 'eqip', 'version.py')) as fp:
    exec(fp.read(), version)

def get_appropriate_version(version, revision):
    if 'dev' in version:
        commit_date = subprocess.check_output(['git', 'log', '-1', '--format=%ci', revision]).strip().decode('ascii')
        return '%s_%s_%s_%s' % (
            version,
            commit_date.split(' ')[0],
            commit_date.split(' ')[1].replace(':', '-'),
            revision
        )
    return version
    
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii') if args.revision is None else args.revision
version  = get_appropriate_version(version['__version__'], revision) if args.version is None else args.version

if args.name is None:
    tag = []
elif version is None:
    tag = ['-t', '%s/%s-py%s' % (args.repository, args.name, args.python)]
else:
    tag = ['-t', '%s/%s:%s-py%s' % (args.repository, args.name, version, args.python)]



docker_cmd = [] + \
  ['docker', 'build', '--build-arg', 'NUM_MAKE_CORES=%s' % args.num_make_jobs, '--build-arg', 'EQIP_REVISION=%s' % revision] + \
  tag + \
  [os.path.join(here, 'docker-container', 'python%s' % args.python)]

print(docker_cmd)

subprocess.call(docker_cmd)





