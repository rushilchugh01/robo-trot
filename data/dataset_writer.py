import sys

from robo_trot.data_pipeline import dataset_writer as _impl

sys.modules[__name__] = _impl
