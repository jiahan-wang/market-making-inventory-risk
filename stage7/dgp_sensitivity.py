"""Re-run the selected stage 7 policy under a different order-flow generator.

Same final-split seeds, same policy, with the informed share and the informed
accuracy moved. It separates the part of the result that belongs to the policy from
the part that belongs to the generator. Nothing here feeds back into selection.

Run from the repository root: python stage7/dgp_sensitivity.py
"""
from __future__ import annotations

import contextlib
import csv
from dataclasses import replace
from pathlib import Path

from experiment import markets_for, write_csv
from liquidity_budget_optimizer import TEST_SEEDS, feasible, liquidity_summary
from risk_budget_optimizer import evaluate_policy
from simulator import Config, Policy, REGIMES

# The share multipliers bracket the shipped 0.10 / 0.40 / 0.70 regime mix; the
# accuracy column spans a much less predictive informed trader up to a perfect one.
SHARE_MULTIPLIERS = (0.6, 0.8, 1.0, 1.2, 1.4)
INFORMED_ACCURACY = (0.70, 0.85, 0.95, 1.00)


@contextlib.contextmanager
def regimes_with(multiplier: float):
    """Swap the regime table for the duration of one cell.

    build_market reads the module-level REGIMES, so a temporary substitution is the
    only way to vary the generator without duplicating it.
    """
    original = dict(REGIMES)
    REGIMES.clear()
    REGIMES.update({name: (sigma, min(0.95, share * multiplier))
                    for name, (sigma, share) in original.items()})
    try:
        yield
    finally:
        REGIMES.clear()
        REGIMES.update(original)


def selected_policy() -> tuple[Config, Policy]:
    """Rebuild the policy stage 7 committed to, from its own committed summary table.

    Reading the table rather than re-typing the parameters keeps this analysis
    attached to whatever stage 7 actually selected.
    """
    with (Path(__file__).resolve().parent / "output" / "test_360_policy_summary.csv").open(
            encoding="utf-8-sig", newline="") as f:
        row = next(r for r in csv.DictReader(f) if r["label"] == "stage7_selected")
    threshold = row["hedge_threshold"]
    policy = Policy(row["strategy"], float(row["base_half_spread"]),
                    float(row["inventory_alpha"]), float(row["volatility_beta"]),
                    None if threshold == "none" else int(threshold))
    return Config(), policy


def run(out_dir: Path | None = None) -> list[dict]:
    cfg, policy = selected_policy()
    out_dir = out_dir or (Path(__file__).resolve().parent / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for multiplier in SHARE_MULTIPLIERS:
        for accuracy in INFORMED_ACCURACY:
            with regimes_with(multiplier):
                cell_cfg = replace(cfg, informed_accuracy=accuracy)
                markets = list(markets_for(cell_cfg, TEST_SEEDS).items())
                stats = liquidity_summary(evaluate_policy(policy, cell_cfg, markets))
            rows.append({
                "informed_share_multiplier": multiplier,
                "informed_accuracy": accuracy,
                "mean_net_pnl": stats["mean_net_pnl"],
                "mean_max_drawdown": stats["mean_max_drawdown"],
                "stressed_mean_max_drawdown": stats["stressed_mean_max_drawdown"],
                "stressed_fill_rate": stats["stressed_fill_rate"],
                "tail_loss_rate": stats["tail_loss_rate"],
                "tail_loss_wilson_upper95": stats["tail_loss_wilson_upper95"],
                "mean_hedge_cost": stats["mean_hedge_cost"],
                "screens_met_on_this_generator": feasible(stats),
            })
            print(f"share x{multiplier:.1f} accuracy {accuracy:.2f}: "
                  f"pnl={rows[-1]['mean_net_pnl']:+.4f} "
                  f"stressDD={rows[-1]['stressed_mean_max_drawdown']:.4f} "
                  f"stressFill={rows[-1]['stressed_fill_rate']:.2%} "
                  f"screens={rows[-1]['screens_met_on_this_generator']}")
    write_csv(out_dir / "dgp_sensitivity.csv", rows)
    passing = sum(1 for r in rows if r["screens_met_on_this_generator"])
    print(f"\nSelected policy: {policy}")
    print(f"Screens met on {passing}/{len(rows)} generator variants, "
          f"mean net P&L range {min(r['mean_net_pnl'] for r in rows):+.4f} "
          f"to {max(r['mean_net_pnl'] for r in rows):+.4f}.")
    print(f"Wrote {out_dir / 'dgp_sensitivity.csv'}")
    return rows


if __name__ == "__main__":
    run()
