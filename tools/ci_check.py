"""Regenerate the stage 7 tables into a scratch directory and compare them with the
committed copies. Run from the repository root: python tools/ci_check.py

Text cells have to match exactly. Numeric cells are compared with a tolerance,
because the price paths come from random.gauss(), which goes through libm, and
CPython 3.12 changed builtin sum() for floats. Either one moves the last bits of
every aggregate without moving a single trading decision. The round-level ledger
is the evidence for that: it is byte-identical across platforms, while the
summary tables differ in the fifteenth digit.

Pass --strict to demand byte equality. That holds on the machine that generated
the tables and is the check worth running before you commit new numbers.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage7"))

from liquidity_budget_optimizer import run  # noqa: E402

ABS_TOL = 1e-9
REL_TOL = 1e-9


def read_table(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1:]


def compare_numeric(path_a: Path, path_b: Path):
    """Return (header list, worst |a-b|) or None when the shapes disagree."""
    head_a, rows_a = read_table(path_a)
    head_b, rows_b = read_table(path_b)
    if head_a != head_b or len(rows_a) != len(rows_b):
        return None
    worst = 0.0
    for row_a, row_b in zip(rows_a, rows_b):
        for cell_a, cell_b in zip(row_a, row_b):
            if cell_a == cell_b:
                continue
            try:
                value_a, value_b = float(cell_a), float(cell_b)
            except ValueError:
                return None
            gap = abs(value_a - value_b)
            if gap > ABS_TOL + REL_TOL * max(abs(value_a), abs(value_b)):
                return None
            worst = max(worst, gap)
    return head_a, worst


def main(argv):
    strict = "--strict" in argv
    scratch = ROOT / ".ci_repro"
    run(scratch)
    shipped = ROOT / "stage7" / "output"

    bad = []
    worst_overall = 0.0
    for fresh in sorted(scratch.iterdir()):
        old = shipped / fresh.name
        if not old.is_file():
            print("not committed:", fresh.name)
            bad.append(fresh.name)
            continue
        if fresh.suffix == ".csv":
            result = compare_numeric(fresh, old)
            if result is None:
                print("DIFFERS  ", fresh.name)
                bad.append(fresh.name)
                continue
            header, worst = result
            worst_overall = max(worst_overall, worst)
            if strict and worst > 0.0:
                print(f"DIFFERS   {fresh.name}  (max |delta| {worst:.3e})")
                bad.append(fresh.name)
        elif fresh.read_bytes() != old.read_bytes():
            print("DIFFERS  ", fresh.name)
            bad.append(fresh.name)

    ok = not bad
    if ok and strict:
        print("stage 7 reproduction is byte-identical: True")
    elif ok:
        print(f"stage 7 reproduction matches to 1e-9: True  "
              f"(worst numeric delta {worst_overall:.3e})")
    else:
        print("stage 7 reproduction is byte-identical:", False)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
