"""Stage 6: stage 5's simulator with terminal liquidation on the equity path.

Closing the surviving position costs liquidation_half_spread plus a fee, and that
entry falls on both net P&L and drawdown so the two numbers stay on one basis.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
import random
from statistics import mean

REGIMES = {
    "calm": (0.025, 0.10),
    "normal": (0.060, 0.40),
    "stressed": (0.120, 0.70),
}


@dataclass(frozen=True)
class Config:
    rounds: int = 150
    initial_mid: float = 100.0
    customer_max_edge: float = 0.16
    informed_accuracy: float = 0.95
    max_inventory: int = 5
    fee_per_trade: float = 0.005
    liquidation_half_spread: float = 0.05
    vol_window: int = 15
    vol_reference: float = 0.060
    min_half_spread: float = 0.018
    max_half_spread: float = 0.22

    def validate(self):
        if self.rounds < 1 or self.max_inventory < 1 or self.vol_window < 1:
            raise ValueError("rounds, max_inventory, and vol_window must be positive")
        if self.initial_mid <= 0 or self.customer_max_edge <= 0:
            raise ValueError("initial_mid and customer_max_edge must be positive")
        if not 0.5 <= self.informed_accuracy <= 1:
            raise ValueError("informed_accuracy must be in [0.5, 1]")
        if min(self.fee_per_trade, self.liquidation_half_spread, self.vol_reference) < 0:
            raise ValueError("costs and vol_reference must be nonnegative")
        if not 0 < self.min_half_spread < self.max_half_spread:
            raise ValueError("invalid spread bounds")


@dataclass(frozen=True)
class Event:
    index: int
    mid_before: float
    customer_side: str
    customer_limit: float
    mid_after: float
    informed: bool  # hidden from all strategies; visible only to evaluation


@dataclass(frozen=True)
class Policy:
    name: str
    base_half_spread: float
    inventory_alpha: float = 0.0
    volatility_beta: float = 0.0

    def validate(self, cfg: Config):
        if self.base_half_spread <= 0 or self.inventory_alpha < 0 or self.volatility_beta < 0:
            raise ValueError("positive base spread; nonnegative alpha and beta required")
        if not self.name:
            raise ValueError("policy name is required")


def build_market(cfg: Config, regime: str, seed: int) -> list[Event]:
    """Generate all price paths and orders before any policy runs.

    An informed order's direction is a noisy signal of the next change, and is
    never passed to the quoting function.
    """
    cfg.validate()
    if regime not in REGIMES:
        raise ValueError(f"Unknown regime {regime}")
    sigma, informed_share = REGIMES[regime]
    rng = random.Random(seed)
    mid = cfg.initial_mid
    events = []
    for t in range(1, cfg.rounds + 1):
        price_shock = rng.gauss(0, sigma)
        next_mid = max(0.01, mid + price_shock)
        informed = rng.random() < informed_share
        if informed:
            informed_correct = rng.random() < cfg.informed_accuracy
            side = ("buy" if price_shock >= 0 else "sell") if informed_correct else ("sell" if price_shock >= 0 else "buy")
        else:
            side = rng.choice(("buy", "sell"))
        edge = rng.uniform(0, cfg.customer_max_edge)
        limit = mid + edge if side == "buy" else mid - edge
        events.append(Event(t, mid, side, limit, next_mid, informed))
        mid = next_mid
    return events


def recent_volatility(past_changes: list[float], cfg: Config) -> float:
    """Root mean square of prior mid changes; the current change is excluded."""
    if not past_changes:
        return cfg.vol_reference
    window = past_changes[-cfg.vol_window:]
    return sqrt(sum(v * v for v in window) / len(window))


def make_quotes(mid: float, inventory: int, past_vol: float, policy: Policy, cfg: Config) -> tuple[float, float, float]:
    """Inputs are the current mid, own inventory and past realized volatility."""
    policy.validate(cfg)
    center = mid - policy.inventory_alpha * inventory
    width = min(cfg.max_half_spread,
                max(cfg.min_half_spread,
                    policy.base_half_spread + policy.volatility_beta * (past_vol - cfg.vol_reference)))
    bid = round(center - width, 5)
    ask = round(center + width, 5)
    if not (0 < bid < ask):
        raise ValueError("invalid quote")
    return bid, ask, width


def simulate(events: list[Event], cfg: Config, policy: Policy, *, record_rounds: bool = False) -> tuple[dict, list[dict]]:
    """Run one pre-generated market. Quotes depend on inventory; fills do not move prices.

    A customer order's side, limit and informed flag are not passed to make_quotes.
    All customer orders are single-unit and only one order arrives per round.
    """
    cfg.validate()
    policy.validate(cfg)
    if len(events) != cfg.rounds:
        raise ValueError("event count mismatch")
    cash = 0.0
    inventory = 0
    fills = 0
    informed_fills = 0
    fees = 0.0
    abs_inventory_sum = 0.0
    max_abs_inventory = 0
    peak_equity = 0.0
    max_drawdown = 0.0
    past_changes: list[float] = []
    markouts_informed = []
    markouts_noise = []
    rows = []
    for ev in events:
        vol = recent_volatility(past_changes, cfg)
        q_before = inventory
        bid, ask, width = make_quotes(ev.mid_before, q_before, vol, policy, cfg)
        wants_buy = ev.customer_side == "buy"
        eligible = (ask <= ev.customer_limit) if wants_buy else (bid >= ev.customer_limit)
        within_cap = (inventory > -cfg.max_inventory) if wants_buy else (inventory < cfg.max_inventory)
        filled = eligible and within_cap
        price = ask if wants_buy else bid
        old_cash = cash
        if filled:
            fees += cfg.fee_per_trade
            fills += 1
            informed_fills += int(ev.informed)
            if wants_buy:
                cash += price - cfg.fee_per_trade
                inventory -= 1
                markout = price - ev.mid_after
            else:
                cash -= price + cfg.fee_per_trade
                inventory += 1
                markout = ev.mid_after - price
            (markouts_informed if ev.informed else markouts_noise).append(markout)
        assert abs(inventory) <= cfg.max_inventory
        inventory_delta = (-1 if wants_buy else 1) if filled else 0
        cash_delta = (price - cfg.fee_per_trade if wants_buy else -price - cfg.fee_per_trade) if filled else 0.0
        assert inventory == q_before + inventory_delta
        assert abs(cash - (old_cash + cash_delta)) < 1e-8
        equity = cash + inventory * ev.mid_after
        peak_equity = max(peak_equity, equity)
        max_drawdown = max(max_drawdown, peak_equity - equity)
        abs_inventory_sum += abs(inventory)
        max_abs_inventory = max(max_abs_inventory, abs(inventory))
        if record_rounds:
            rows.append({
                "round": ev.index, "strategy": policy.name,
                "mid_before": round(ev.mid_before, 5),
                "past_vol_estimate": round(vol, 5),
                "inventory_before": q_before,
                "quote_half_spread": round(width, 5),
                "bid": bid, "ask": ask,
                "customer_side": ev.customer_side,
                "customer_limit": round(ev.customer_limit, 5),
                "informed": int(ev.informed),
                "filled": int(filled), "trade_price": price if filled else "",
                "cash_after": round(cash, 5),
                "inventory_after": inventory,
                "mid_after": round(ev.mid_after, 5),
                "equity_after": round(equity, 5),
            })
        # The post-fill mid joins the volatility window only from the next round on.
        past_changes.append(ev.mid_after - ev.mid_before)
    marked_equity = cash + inventory * events[-1].mid_after
    liquidate_cost = abs(inventory) * (cfg.liquidation_half_spread + cfg.fee_per_trade)
    # Close the remaining position at the assumed liquidation spread plus fee. It is a
    # final point on the equity path, so max drawdown and net P&L share one basis.
    net_pnl = marked_equity - liquidate_cost
    pre_liquidation_max_drawdown = max_drawdown
    peak_equity = max(peak_equity, net_pnl)
    max_drawdown = max(max_drawdown, peak_equity - net_pnl)
    if record_rounds:
        rows.append({
            "round": cfg.rounds + 1, "strategy": policy.name,
            "event_kind": "terminal_liquidation",
            "mid_before": round(events[-1].mid_after, 5),
            "inventory_before": inventory,
            "cash_after": round(net_pnl, 5),
            "inventory_after": 0,
            "liquidation_cost": round(liquidate_cost, 5),
            "equity_after": round(net_pnl, 5),
        })
    return {
        "strategy": policy.name, "rounds": cfg.rounds, "fills": fills,
        "fill_rate": fills / cfg.rounds,
        "informed_fills": informed_fills,
        "informed_fill_share": informed_fills / fills if fills else 0,
        "avg_abs_inventory": abs_inventory_sum / cfg.rounds,
        "max_abs_inventory": max_abs_inventory,
        "ending_inventory": inventory,
        "fees": fees,
        "marked_equity": marked_equity,
        "liquidation_cost": liquidate_cost,
        "net_pnl": net_pnl,
        "max_drawdown": max_drawdown,
        "max_drawdown_before_liquidation": pre_liquidation_max_drawdown,
        "mean_informed_markout": mean(markouts_informed) if markouts_informed else float("nan"),
        "mean_noise_markout": mean(markouts_noise) if markouts_noise else float("nan"),
    }, rows


def starter_policies() -> list[Policy]:
    return [
        Policy("A_fixed", 0.065),
        Policy("B_inventory", 0.065, inventory_alpha=0.012),
        Policy("C_vol_adaptive", 0.065, inventory_alpha=0.012, volatility_beta=0.85),
    ]
