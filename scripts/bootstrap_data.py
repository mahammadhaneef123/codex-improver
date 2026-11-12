"""Generate the development dataset into CSV."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codex_maint.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["generate-data", *sys.argv[1:]]))
