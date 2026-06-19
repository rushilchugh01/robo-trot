import sys

from robo_trot.demos import record_teacher_demos as _impl


if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
