from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.policy.audit_action_mapping import *  # noqa: F401,F403
from scripts.policy.audit_action_mapping import main


if __name__ == "__main__":
    main()
