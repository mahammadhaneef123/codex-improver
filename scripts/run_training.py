"""Train the model and run a monitoring pass in one command."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codex_maint.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["run-all", *sys.argv[1:]]))
