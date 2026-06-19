import sys

from robo_trot.demos import dataset_writer as _impl

sys.modules[__name__] = _impl
