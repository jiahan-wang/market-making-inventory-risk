"""Market generator and CSV writer for stage 4. Imported, never run directly.

Cut down from stage3/experiment.py to the pieces this stage still needs. Paths are
generated before any policy runs, so every strategy sees the same markets.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
import csv

from simulator import Config, Policy, build_market, REGIMES

REGIME_NAMES = tuple(REGIMES)
TRAIN_SEEDS = tuple(range(20000, 20030))
TEST_SEEDS = tuple(range(40000, 40120))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    if not rows:
        raise ValueError("empty table")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(rows)


def markets_for(cfg: Config, seeds: tuple[int, ...]) -> dict[tuple[str, int], list]:
    # One PRNG stream per (seed, regime) pair: the key is a bijection, so no
    # stream can be shared between two splits however the seed ranges are chosen.
    # Within one market every strategy sees the same path.
    return {(reg, seed): build_market(cfg, reg, seed * len(REGIME_NAMES) + ri)
            for ri, reg in enumerate(REGIME_NAMES) for seed in seeds}


def candidates(group: str) -> list[Policy]:
    spreads = (0.05, 0.07, 0.09)
    if group == "A":
        return [Policy("A_fixed", d) for d in spreads]
    if group == "B":
        return [Policy("B_inventory", d, a) for d, a in product(spreads, (0.008, 0.016, 0.024))]
    if group == "C":
        return [Policy("C_vol_adaptive", d, a, b) for d, a, b in
                product(spreads, (0.008, 0.016, 0.024), (0.4, 1.0, 1.6))]
    raise ValueError(group)


def evaluate(markets: dict, cfg: Config, policies: list[Policy]) -> list[dict]:
    from simulator import simulate
    records = []
    for (regime, seed), path in markets.items():
        for p in policies:
            stats, _ = simulate(path, cfg, p)
            records.append({"regime": regime, "seed": seed, **stats})
    return records
