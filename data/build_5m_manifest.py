import sys

from robo_trot.data_pipeline import manifest as _impl


if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
