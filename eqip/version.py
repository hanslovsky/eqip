import sys

__major__   = 0
__minor__   = 4
__patch__   = 2
__tag__     = 'dev'
# Unfortunately, < 3.5 does not support literal format strings -- too bad!
# if sys.version_info[0] >= 3 and sys.version_info[1] >= 5:
#     __version__ = f'{__major__}.{__minor__}.{__patch__}{__tag__}'
# else:
__version__ = '{}.{}.{}.{}'.format(__major__, __minor__, __patch__, __tag__).strip('.')
