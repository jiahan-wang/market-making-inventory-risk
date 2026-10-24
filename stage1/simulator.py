"""Stage 1: one fixed spread, one customer per round, a random-walk mid.

Run it directly (python simulator.py); it rewrites output/trades_100_rounds.csv.
Amounts are simulated price units throughout.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    n_rounds: int = 100
    seed: int = 2026
    initial_mid: float = 100.00
    half_spread: float = 0.05
    price_sigma: float = 0.03
    max_customer_edge: float = 0.10
    max_inventory: int = 5
    fee_per_trade: float = 0.005


@dataclass
class Trader:
    cash: float = 0.0
    inventory: int = 0
    total_fees: float = 0.0

    def account_equity(self, mark: float) -> float:
        """Mark-to-market equity; excludes the cost of closing the final position."""
        return self.cash + self.inventory * mark

    def accept_trade(self, side: str, price: float, fee: float) -> None:
        """Side is the customer's side, so a customer buy reduces trader inventory."""
        if side == "buy":
            self.cash += price - fee
            self.inventory -= 1
        elif side == "sell":
            self.cash -= price + fee
            self.inventory += 1
        else:
            raise ValueError(f"Unknown side: {side}")
        self.total_fees += fee


def make_quotes(mid: float, cfg: Config) -> tuple[float, float]:
    """Symmetric quotes around the mid; inventory and future prices are not inputs."""
    return round(mid - cfg.half_spread, 4), round(mid + cfg.half_spread, 4)


def simulate(cfg: Config = Config()) -> tuple[list[dict], Trader, dict]:
    if cfg.n_rounds < 1 or cfg.max_inventory < 1 or cfg.half_spread <= 0:
        raise ValueError("n_rounds, max_inventory, and half_spread must be positive")
    if cfg.price_sigma < 0 or cfg.max_customer_edge <= 0 or cfg.fee_per_trade < 0:
        raise ValueError("price_sigma, max_customer_edge, and fee_per_trade invalid")

    rng = random.Random(cfg.seed)
    mid = cfg.initial_mid
    trader = Trader()
    rows: list[dict] = []
    n_fills = 0
    n_eligible = 0
    total_abs_inventory = 0

    for t in range(1, cfg.n_rounds + 1):
        # Quote on the mid that is public at this point in the loop.
        current_mid = mid
        bid, ask = make_quotes(current_mid, cfg)
        prev_cash = trader.cash
        prev_inventory = trader.inventory

        # One customer arrives; its reservation price is not observable at quote time.
        customer_side = rng.choice(["buy", "sell"])
        customer_edge = rng.uniform(0, cfg.max_customer_edge)
        customer_limit = (
            current_mid + customer_edge if customer_side == "buy"
            else current_mid - customer_edge
        )
        eligible = (
            ask <= customer_limit if customer_side == "buy"
            else bid >= customer_limit
        )
        n_eligible += int(eligible)

        # The inventory cap blocks further same-direction fills.
        permitted = (
            trader.inventory > -cfg.max_inventory if customer_side == "buy"
            else trader.inventory < cfg.max_inventory
        )
        filled = eligible and permitted
        execution_price = ask if customer_side == "buy" else bid
        if filled:
            trader.accept_trade(customer_side, execution_price, cfg.fee_per_trade)
            n_fills += 1
        assert abs(trader.inventory) <= cfg.max_inventory
        assert round(trader.cash - prev_cash, 8) == round(
            ((execution_price - cfg.fee_per_trade) if customer_side == "buy"
             else (-execution_price - cfg.fee_per_trade)) if filled else 0.0, 8
        )
        assert trader.inventory - prev_inventory == (
            (-1 if customer_side == "buy" else 1) if filled else 0
        )

        # The mid moves after the fill, so it can only affect the next round.
        next_mid = max(0.01, current_mid + rng.gauss(0, cfg.price_sigma))
        mid = next_mid
        total_abs_inventory += abs(trader.inventory)
        rows.append({
            "round": t,
            "mid_before": round(current_mid, 4),
            "bid": bid,
            "ask": ask,
            "customer_side": customer_side,
            "customer_limit": round(customer_limit, 4),
            "price_acceptable": int(eligible),
            "inventory_permits": int(permitted),
            "filled": int(filled),
            "trade_price": round(execution_price, 4) if filled else "",
            "cash_after": round(trader.cash, 4),
            "inventory_after": trader.inventory,
            "mid_after": round(next_mid, 4),
            "equity_after": round(trader.account_equity(next_mid), 4),
        })

    # Marked equity, before the cost of closing the remaining position.
    stats = {
        "rounds": cfg.n_rounds,
        "filled": n_fills,
        "fill_rate": n_fills / cfg.n_rounds,
        "eligible": n_eligible,
        "ending_mid": mid,
        "ending_cash": trader.cash,
        "ending_inventory": trader.inventory,
        "total_fees": trader.total_fees,
        "ending_equity_marked": trader.account_equity(mid),
        "average_abs_inventory": total_abs_inventory / cfg.n_rounds,
    }
    return rows, trader, stats


def write_csv(rows: list[dict], filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cfg = Config()
    rows, trader, stats = simulate(cfg)
    file_path = Path(__file__).resolve().parent / "output" / "trades_100_rounds.csv"
    write_csv(rows, file_path)

    print("=== Stage 1 / Fixed-spread market maker ===")
    print(f"Seed: {cfg.seed} | rounds: {stats['rounds']} | quote: mid ± {cfg.half_spread:.2f}")
    print("\nFirst 10 rounds:")
    print(f"{'t':>3} {'mid':>9} {'bid':>9} {'ask':>9} {'side':>6} {'fill':>5} {'inventory':>10} {'equity':>11}")
    for r in rows[:10]:
        print(f"{r['round']:>3} {r['mid_before']:>9.4f} {r['bid']:>9.4f} {r['ask']:>9.4f} "
              f"{r['customer_side']:>6} {r['filled']:>5} {r['inventory_after']:>10} {r['equity_after']:>11.4f}")

    print("\nSummary:")
    print(f"Filled trades: {stats['filled']}/{stats['rounds']} ({100 * stats['fill_rate']:.1f}%)")
    print(f"Average |inventory|: {stats['average_abs_inventory']:.2f} units")
    print(f"Ending inventory: {stats['ending_inventory']} units")
    print(f"Ending cash: {stats['ending_cash']:.4f}")
    print(f"Ending market mid: {stats['ending_mid']:.4f}")
    print(f"Total trade fees: {stats['total_fees']:.4f}")
    print(f"Ending marked equity: {stats['ending_equity_marked']:.4f}")
    print(f"CSV saved: {file_path}")


if __name__ == "__main__":
    main()
