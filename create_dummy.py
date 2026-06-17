"""Deprecated shim — use scripts/create_test_index.py instead."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "scripts" / "create_test_index.py"
    print("Note: create_dummy.py is deprecated. Running create_test_index.py …")
    cmd = [sys.executable, str(script), "--force", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))