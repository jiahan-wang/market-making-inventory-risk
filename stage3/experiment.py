"""Stage 3: train on one seed range, test on a disjoint one, strategies matched.

Run it directly (python experiment.py). Paths are generated before any policy runs,
so every strategy in a comparison meets the same markets.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import mean, stdev
from math import sqrt
import csv

from simulator import Config, Policy, build_market, simulate, REGIMES

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
        return [Policy("C_vol_adaptive", d, a, b) for d, a, b in product(spreads, (0.008, 0.016, 0.024), (0.4, 1.0, 1.6))]
    raise ValueError(group)


def objective(stats: list[dict]) -> float:
    """One risk-aware criterion, applied unchanged to all three groups.

Units are simulated price units.
"""
    return mean(s["net_pnl"] - 0.8 * s["max_drawdown"] - 0.04 * s["avg_abs_inventory"] for s in stats)


def fit_policies(cfg: Config, markets: dict) -> tuple[list[Policy], list[dict]]:
    fitted = []
    ranking = []
    paths = list(markets.values())
    for group in ("A", "B", "C"):
        trials = []
        for p in candidates(group):
            records = [simulate(path, cfg, p)[0] for path in paths]
            score = objective(records)
            trials.append((score, p))
            ranking.append({"group": group, "base_half_spread": p.base_half_spread,
                            "inventory_alpha": p.inventory_alpha,
                            "volatility_beta": p.volatility_beta,
                            "training_objective": score})
        best_score, winner = max(trials, key=lambda t: t[0])
        fitted.append(winner)
    return fitted, ranking


def evaluate(markets: dict, cfg: Config, policies: list[Policy]) -> list[dict]:
    records = []
    for (regime, seed), path in markets.items():
        for p in policies:
            stats, _ = simulate(path, cfg, p)
            records.append({"regime": regime, "seed": seed, **stats})
    return records


def summaries(records: list[dict]) -> list[dict]:
    table = []
    for regime in (*REGIME_NAMES, "all"):
        for strategy in ("A_fixed", "B_inventory", "C_vol_adaptive"):
            matched = [r for r in records if r["strategy"] == strategy and (regime == "all" or r["regime"] == regime)]
            pnl = [r["net_pnl"] for r in matched]
            table.append({
                "regime": regime, "strategy": strategy,
                "scenarios": len(matched),
                "avg_net_pnl": mean(pnl),
                "sd_net_pnl": stdev(pnl),
                "positive_pnl_rate": mean(float(x > 0) for x in pnl),
                "avg_fill_rate": mean(r["fill_rate"] for r in matched),
                "avg_abs_inventory": mean(r["avg_abs_inventory"] for r in matched),
                "avg_max_drawdown": mean(r["max_drawdown"] for r in matched),
                "avg_informed_fill_share": mean(r["informed_fill_share"] for r in matched),
                "avg_informed_markout": mean(r["mean_informed_markout"] for r in matched if r["mean_informed_markout"] == r["mean_informed_markout"]),
                "avg_noise_markout": mean(r["mean_noise_markout"] for r in matched if r["mean_noise_markout"] == r["mean_noise_markout"]),
            })
    return table


def paired_interval(records: list[dict], newer: str, older: str, metric: str, regime: str = "all") -> dict:
    groups = defaultdict(dict)
    for r in records:
        if regime == "all" or r["regime"] == regime:
            groups[(r["regime"], r["seed"])][r["strategy"]] = r
    diffs = [v[newer][metric] - v[older][metric] for v in groups.values()]
    mu = mean(diffs)
    err = 1.96 * stdev(diffs) / sqrt(len(diffs))
    return {"regime": regime, "comparison": f"{newer} minus {older}", "metric": metric,
            "pairs": len(diffs), "mean_difference": mu, "lower_95_approx": mu-err, "upper_95_approx": mu+err}


def run(cfg: Config = Config(), *, train_seeds=TRAIN_SEEDS, test_seeds=TEST_SEEDS):
    if set(train_seeds) & set(test_seeds):
        raise ValueError("Training and test seeds must not overlap")
    train_markets = markets_for(cfg, tuple(train_seeds))
    policies, ranking = fit_policies(cfg, train_markets)
    test_markets = markets_for(cfg, tuple(test_seeds))
    records = evaluate(test_markets, cfg, policies)
    return policies, ranking, records, summaries(records)


def main():
    cfg = Config()
    policies, ranking, records, table = run(cfg)
    output = Path(__file__).resolve().parent / "output"
    write_csv(output / "training_grid.csv", ranking)
    write_csv(output / "test_360_market_results.csv", records)
    write_csv(output / "test_summary.csv", table)
    comparisons = []
    for regime in (*REGIME_NAMES, "all"):
        for newer, older in (("B_inventory", "A_fixed"), ("C_vol_adaptive", "B_inventory"), ("C_vol_adaptive", "A_fixed")):
            for metric in ("net_pnl", "avg_abs_inventory", "max_drawdown", "fill_rate"):
                comparisons.append(paired_interval(records, newer, older, metric, regime))
    write_csv(output / "paired_intervals.csv", comparisons)
    demonstration = build_market(cfg, "stressed", 2026)
    demo_rows = []
    for p in policies:
        _, rows = simulate(demonstration, cfg, p, record_rounds=True)
        demo_rows.extend(rows)
    write_csv(output / "one_stressed_market_rounds.csv", demo_rows)
    lines = [
        "Stage 3: quoting under adverse selection with adaptive spreads",
        f"Training: {len(TRAIN_SEEDS)} independent seeds per regime x {len(REGIME_NAMES)} regimes; separate tuning per strategy family",
        f"Final split: {len(TEST_SEEDS)} seeds per regime x {len(REGIME_NAMES)} regimes = {len(TEST_SEEDS)*len(REGIME_NAMES)} matched markets; {cfg.rounds} rounds/market",
        "Order flow is synthetic; intervals describe this generator only.",
        "Selected policies (shared training objective):",
    ]
    for p in policies:
        lines.append(f"  {p.name}: base_half_spread={p.base_half_spread}, inventory_alpha={p.inventory_alpha}, volatility_beta={p.volatility_beta}")
    lines.append("\nFinal split: aggregation across regimes (equal weighting):")
    for s in [r for r in table if r["regime"] == "all"]:
        lines.append(f"  {s['strategy']}: net_pnl={s['avg_net_pnl']:.4f} (sd={s['sd_net_pnl']:.4f}), |inventory|={s['avg_abs_inventory']:.3f}, maxDD={s['avg_max_drawdown']:.3f}, fill rate={s['avg_fill_rate']:.1%}")
    lines.append("\nStressed regime (informed share 0.70, price sigma 0.12):")
    for s in [r for r in table if r["regime"] == "stressed"]:
        lines.append(f"  {s['strategy']}: net_pnl={s['avg_net_pnl']:.4f}, mean informed markout={s['avg_informed_markout']:.4f}, noise markout={s['avg_noise_markout']:.4f}, fill={s['avg_fill_rate']:.1%}")
    for comparison in comparisons:
        if comparison["regime"] == "all" and comparison["metric"] in ("net_pnl", "max_drawdown"):
            lines.append(f"  {comparison['comparison']} ({comparison['metric']}): difference {comparison['mean_difference']:.4f}, approximate 95% interval [{comparison['lower_95_approx']:.4f}, {comparison['upper_95_approx']:.4f}]")
    text = "\n".join(lines) + "\n"
    (output / "run_summary.txt").write_text(text, encoding='utf-8', newline='\n')
    print(text)


if __name__ == "__main__":
    main()
