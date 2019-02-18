import sys

__major__   = 0
__minor__   = 4
__patch__   = 5
__tag__     = 'dev'
__version__ = f'{__major__}.{__minor__}.{__patch__}.{__tag__}'.strip('.')

class _Version(object):

    def major(self):
        return __major__

    def minor(self):
        return __minor__

    def patch(self):
        return __patch__

    def tag(self):
        return __tag__

    def version(self):
        return __version__

    def __str__(self):
        return self.version()

_version = _Version()
