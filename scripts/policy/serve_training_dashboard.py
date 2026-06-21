from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robo_trot.training.dashboard import serve_dashboard


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the dashboard server.

    The returned namespace is consumed by the standalone script entry point.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the training dashboard HTTP server.

    This is the direct execution entry point for the script wrapper.
    """
    args = parse_args(argv)
    serve_dashboard(args.run_dir, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
