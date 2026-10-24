"""Stage 2: fixed spreads against inventory-aware spreads, same markets for both.

Run it directly (python simulator.py) to rewrite output/. Demand and prices are
synthetic, and in this stage a fill does not move the price.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from math import sqrt


@dataclass(frozen=True)
class Config:
    n_rounds: int = 100
    seed: int = 2026
    initial_mid: float = 100.0
    half_spread: float = 0.05
    inventory_alpha: float = 0.01
    price_sigma: float = 0.03
    max_customer_edge: float = 0.10
    max_inventory: int = 5
    fee_per_trade: float = 0.005
    liquidation_half_spread: float = 0.05

    def validate(self) -> None:
        if self.n_rounds < 1 or self.max_inventory < 1 or self.initial_mid <= 0:
            raise ValueError("n_rounds, max_inventory and initial_mid must be positive")
        if self.half_spread <= 0 or self.price_sigma < 0 or self.max_customer_edge <= 0:
            raise ValueError("half_spread and max_customer_edge must be positive; price_sigma nonnegative")
        if self.inventory_alpha < 0 or self.fee_per_trade < 0 or self.liquidation_half_spread < 0:
            raise ValueError("inventory_alpha and costs must be nonnegative")
        # Skew must stay below the half-spread so that bid < ask always holds.
        if self.inventory_alpha * self.max_inventory > self.half_spread + 1e-12:
            raise ValueError("inventory_alpha * max_inventory must be <= half_spread for this model")


@dataclass(frozen=True)
class MarketEvent:
    round: int
    mid_before: float
    customer_side: str
    customer_limit: float
    mid_after: float


def generate_market(cfg: Config) -> list[MarketEvent]:
    """Pre-generate all exogenous events, independent of any strategy.

    Customer limits and the next mid are not passed to make_quotes.
    """
    cfg.validate()
    rng = random.Random(cfg.seed)
    mid = cfg.initial_mid
    events = []
    for t in range(1, cfg.n_rounds + 1):
        side = rng.choice(["buy", "sell"])
        edge = rng.uniform(0, cfg.max_customer_edge)
        customer_limit = mid + edge if side == "buy" else mid - edge
        next_mid = max(0.01, mid + rng.gauss(0, cfg.price_sigma))
        events.append(MarketEvent(t, mid, side, customer_limit, next_mid))
        mid = next_mid
    return events


@dataclass
class Trader:
    cash: float = 0.0
    inventory: int = 0
    total_fees: float = 0.0

    def equity(self, mid: float) -> float:
        return self.cash + self.inventory * mid

    def fill(self, customer_side: str, execution_price: float, fee: float) -> None:
        # Customer buy => trader sells; customer sell => trader buys.
        if customer_side == "buy":
            self.cash += execution_price - fee
            self.inventory -= 1
        elif customer_side == "sell":
            self.cash -= execution_price + fee
            self.inventory += 1
        else:
            raise ValueError(f"Invalid side: {customer_side}")
        self.total_fees += fee


STRATEGIES = ("fixed", "inventory_aware")


def make_quotes(mid: float, inventory: int, strategy: str, cfg: Config) -> tuple[float, float]:
    """Quote from the current mid and own inventory."""
    if strategy == "fixed":
        center = mid
    elif strategy == "inventory_aware":
        center = mid - cfg.inventory_alpha * inventory
    else:
        raise ValueError(f"Invalid strategy {strategy}")
    bid, ask = round(center - cfg.half_spread, 4), round(center + cfg.half_spread, 4)
    if not (0 < bid < ask):
        raise ValueError("Quotes must be positive and bid < ask")
    return bid, ask


def simulate(events: list[MarketEvent], cfg: Config, strategy: str) -> tuple[list[dict], dict]:
    cfg.validate()
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    if len(events) != cfg.n_rounds:
        raise ValueError("Event count must equal cfg.n_rounds")
    trader = Trader()
    rows: list[dict] = []
    fills = 0
    total_abs_inventory = 0
    equity_path = [0.0]

    for event in events:
        bid, ask = make_quotes(event.mid_before, trader.inventory, strategy, cfg)
        prev_inventory = trader.inventory
        prev_cash = trader.cash
        customer_buy = event.customer_side == "buy"
        eligible = (ask <= event.customer_limit) if customer_buy else (bid >= event.customer_limit)
        permitted = (trader.inventory > -cfg.max_inventory) if customer_buy else (trader.inventory < cfg.max_inventory)
        filled = eligible and permitted
        price = ask if customer_buy else bid
        if filled:
            trader.fill(event.customer_side, price, cfg.fee_per_trade)
            fills += 1
        assert abs(trader.inventory) <= cfg.max_inventory
        assert trader.inventory - prev_inventory == ((-1 if customer_buy else 1) if filled else 0)
        expected_change = (price - cfg.fee_per_trade) if customer_buy else (-price - cfg.fee_per_trade)
        assert abs((trader.cash - prev_cash) - (expected_change if filled else 0)) < 1e-8
        total_abs_inventory += abs(trader.inventory)
        marked = trader.equity(event.mid_after)
        equity_path.append(marked)
        rows.append({
            "round": event.round,
            "strategy": strategy,
            "mid_before": round(event.mid_before, 4),
            "inventory_before": prev_inventory,
            "bid": bid,
            "ask": ask,
            "customer_side": event.customer_side,
            "customer_limit": round(event.customer_limit, 4),
            "price_acceptable": int(eligible),
            "inventory_permits": int(permitted),
            "filled": int(filled),
            "trade_price": price if filled else "",
            "cash_after": round(trader.cash, 4),
            "inventory_after": trader.inventory,
            "mid_after": round(event.mid_after, 4),
            "equity_after": round(marked, 4),
        })

    final_mid = events[-1].mid_after
    ending_marked = trader.equity(final_mid)
    # Closing the remaining position: adverse price plus per-unit fee, at end only.
    liquidation_cost = abs(trader.inventory) * (cfg.liquidation_half_spread + cfg.fee_per_trade)
    ending_after_liquidation = ending_marked - liquidation_cost
    running_peak = 0.0
    max_drawdown = 0.0
    for equity in equity_path:
        running_peak = max(running_peak, equity)
        max_drawdown = max(max_drawdown, running_peak - equity)
    stats = {
        "strategy": strategy,
        "seed": cfg.seed,
        "rounds": cfg.n_rounds,
        "filled": fills,
        "fill_rate": fills / cfg.n_rounds,
        "ending_inventory": trader.inventory,
        "avg_abs_inventory": total_abs_inventory / cfg.n_rounds,
        "total_fees": trader.total_fees,
        "ending_equity_marked": ending_marked,
        "liquidation_cost": liquidation_cost,
        "ending_pnl_after_liquidation": ending_after_liquidation,
        "max_drawdown": max_drawdown,
    }
    return rows, stats


def compare_once(cfg: Config) -> tuple[list[dict], list[dict]]:
    """Same markets for both strategies."""
    events = generate_market(cfg)
    all_rows = []
    stats = []
    for strategy in STRATEGIES:
        rows, summary = simulate(events, cfg, strategy)
        all_rows.extend(rows)
        stats.append(summary)
    return all_rows, stats


def repeated_comparison(cfg: Config, n_scenarios: int = 200, seed_start: int = 10000) -> tuple[list[dict], list[dict]]:
    """Run fixed policy parameters on a fresh set of markets.

    These are robustness checks over the generator.
    """
    if n_scenarios < 2:
        raise ValueError("At least two scenarios needed")
    from dataclasses import replace
    records = []
    for offset in range(n_scenarios):
        test_cfg = replace(cfg, seed=seed_start + offset)
        _, stats = compare_once(test_cfg)
        records.extend(stats)

    table = []
    for strategy in STRATEGIES:
        matching = [r for r in records if r["strategy"] == strategy]
        pnls = [r["ending_pnl_after_liquidation"] for r in matching]
        table.append({
            "strategy": strategy,
            "scenarios": n_scenarios,
            "mean_pnl_after_liquidation": mean(pnls),
            "std_pnl_after_liquidation": stdev(pnls),
            "fraction_positive_pnl": sum(v > 0 for v in pnls) / len(pnls),
            "mean_fill_rate": mean(r["fill_rate"] for r in matching),
            "mean_abs_inventory": mean(r["avg_abs_inventory"] for r in matching),
            "mean_max_drawdown": mean(r["max_drawdown"] for r in matching),
        })
    return records, table


def paired_summary(records: list[dict]) -> list[dict]:
    """Compare both strategies on matched seed-by-seed markets.

    95% intervals are normal approximations over synthetic market scenarios,
    They describe variation across generated scenarios only.
    """
    by_seed: dict[int, dict[str, dict]] = {}
    for record in records:
        by_seed.setdefault(record["seed"], {})[record["strategy"]] = record
    metrics = ("ending_pnl_after_liquidation", "avg_abs_inventory", "max_drawdown", "fill_rate")
    summary = []
    for metric in metrics:
        differences = [
            by_seed[s]["inventory_aware"][metric] - by_seed[s]["fixed"][metric]
            for s in sorted(by_seed)
        ]
        n = len(differences)
        if n < 2:
            raise ValueError("At least two pairs required")
        avg = mean(differences)
        se = stdev(differences) / sqrt(n)
        summary.append({
            "metric": metric,
            "mean_difference_B_minus_A": avg,
            "se_of_difference": se,
            "lower_95_approx": avg - 1.96 * se,
            "upper_95_approx": avg + 1.96 * se,
            "scenarios": n,
        })
    return summary


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cfg = Config()
    output = Path(__file__).resolve().parent / "output"
    rows, stats = compare_once(cfg)
    per_scenario, summary = repeated_comparison(cfg, n_scenarios=200)
    write_csv(rows, output / "rounds_two_strategies.csv")
    write_csv(stats, output / "one_market_comparison.csv")
    write_csv(per_scenario, output / "all_200_scenarios.csv")
    write_csv(summary, output / "robustness_summary.csv")
    pair_stats = paired_summary(per_scenario)
    write_csv(pair_stats, output / "paired_comparison.csv")
    print("=== Stage 2: fixed vs inventory-aware market making ===")
    print(f"Shared market for both strategies | seed={cfg.seed}, {cfg.n_rounds} rounds")
    print(f"half_spread={cfg.half_spread:.3f}, inventory_alpha={cfg.inventory_alpha:.3f}, cap=±{cfg.max_inventory}")
    print("\nOne market path:")
    print(f"{'Strategy':<18} {'Fills':>5} {'Avg|q|':>8} {'Final q':>8} {'Marked P&L':>13} {'Post-liquidation':>18}")
    for s in stats:
        print(f"{s['strategy']:<18} {s['filled']:>5} {s['avg_abs_inventory']:>8.2f} {s['ending_inventory']:>8} "
              f"{s['ending_equity_marked']:>13.4f} {s['ending_pnl_after_liquidation']:>18.4f}")
    print("\n200 independent markets, parameters fixed in advance:")
    print(f"{'Strategy':<18} {'Mean P&L':>10} {'P&L Std':>10} {'Fill rate':>11} {'Mean |q|':>10} {'Mean max DD':>12}")
    for s in summary:
        print(f"{s['strategy']:<18} {s['mean_pnl_after_liquidation']:>10.4f} {s['std_pnl_after_liquidation']:>10.4f} "
              f"{s['mean_fill_rate']:>11.1%} {s['mean_abs_inventory']:>10.2f} {s['mean_max_drawdown']:>12.4f}")
    pnl_delta = pair_stats[0]
    print(f"\nPaired mean P&L difference (B - A): {pnl_delta['mean_difference_B_minus_A']:+.4f}; "
          f"approx. 95% interval [{pnl_delta['lower_95_approx']:+.4f}, {pnl_delta['upper_95_approx']:+.4f}]")
    print(f"\nCSV files saved to {output}")
    print("Model: prices are independent of orders; no adverse selection or queue effects.")


if __name__ == "__main__":
    main()
