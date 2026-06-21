from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robo_trot.training.parallel_train_bc import main, parse_args


if __name__ == "__main__":
    main()
