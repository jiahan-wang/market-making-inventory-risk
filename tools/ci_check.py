"""Regenerate the stage 7 tables into a scratch directory and diff them against
the committed copies. Run from the repository root: python tools/ci_check.py
"""
from __future__ import annotations

import sys
from filecmp import cmp
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage7"))

from liquidity_budget_optimizer import run  # noqa: E402

scratch = ROOT / ".ci_repro"
run(scratch)
shipped = ROOT / "stage7" / "output"
differs = [f.name for f in sorted(scratch.iterdir())
           if (shipped / f.name).is_file() and not cmp(f, shipped / f.name, shallow=False)]
missing = [f.name for f in sorted(scratch.iterdir()) if not (shipped / f.name).is_file()]
for name in differs:
    print("DIFFERS ", name)
for name in missing:
    print("not committed:", name)
ok = not differs and not missing
print("stage 7 reproduction is byte-identical:", ok)
sys.exit(0 if ok else 1)
