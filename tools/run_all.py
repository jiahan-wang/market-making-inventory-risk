"""Rebuild every artifact in the repository, in dependency order.

    python tools/run_all.py             tables only (standard library)
    python tools/run_all.py --figures   tables, then the matplotlib figures

The stages are independent of each other; the order below is just narrative.
Each one writes into its own output/ directory and is deterministic given its
seed ranges, so a clean run should leave the working tree unchanged.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TABLES = [
    "stage1/simulator.py",
    "stage2/simulator.py",
    "stage3/experiment.py",
    "stage4/risk_preference_experiment.py",
    "stage5/risk_budget_optimizer.py",
    "stage6/stress_budget_optimizer.py",
    "stage7/liquidity_budget_optimizer.py",
    "stage7/dgp_sensitivity.py",
]

FIGURES = [
    "stage2/plot_results.py",
    "stage3/plot_results.py",
    "stage5/plot_results.py",
    "stage6/plot_stage6.py",
    "stage7/plot_stage7.py",
]


def run(script: str) -> int:
    print(f"--- {script}", flush=True)
    return subprocess.call([sys.executable, str(ROOT / script)], cwd=ROOT)


def main(argv: list[str]) -> int:
    scripts = list(TABLES)
    if "--figures" in argv:
        scripts += FIGURES
    failed = [name for name in scripts if run(name) != 0]
    if failed:
        print("failed:", ", ".join(failed))
        return 1
    print(f"{len(scripts)} step(s) completed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
