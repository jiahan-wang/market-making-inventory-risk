"""Checks for the stage 7 simulator, gate and paired comparison."""

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage7"))

from simulator import Config, Event, Policy, build_market, simulate  # noqa: E402
from liquidity_budget_optimizer import (  # noqa: E402
    TRAIN_SEEDS, VALIDATION_SEEDS, TEST_SEEDS, candidate_policies,
    feasible, paired_interval,
)


class Stage7PublicTests(unittest.TestCase):
    def test_seed_reproduces_market_and_splits_are_disjoint(self):
        cfg = Config(rounds=4)
        self.assertEqual(build_market(cfg, "stressed", 123), build_market(cfg, "stressed", 123))
        self.assertTrue(set(TRAIN_SEEDS).isdisjoint(VALIDATION_SEEDS))
        self.assertTrue(set(TRAIN_SEEDS).isdisjoint(TEST_SEEDS))
        self.assertTrue(set(VALIDATION_SEEDS).isdisjoint(TEST_SEEDS))
        self.assertEqual(len(candidate_policies()), 59)

    def test_initial_quote_does_not_use_future_mid_or_hidden_order(self):
        cfg = Config(rounds=1)
        policy = Policy("hedged", 0.05, hedge_threshold=0)
        up = Event(1, 100.0, "buy", 102.0, 101.0, True)
        down = Event(1, 100.0, "sell", 98.0, 99.0, False)
        _, a = simulate([up], cfg, policy, record_rounds=True)
        _, b = simulate([down], cfg, policy, record_rounds=True)
        self.assertEqual((a[0]["bid"], a[0]["ask"]), (b[0]["bid"], b[0]["ask"]))

    def test_hedge_uses_new_mid_and_charges_both_costs(self):
        cfg = Config(rounds=1)
        event = Event(1, 100.0, "buy", 102.0, 101.0, True)
        stat, rows = simulate([event], cfg, Policy("hedged", 0.05, hedge_threshold=0), record_rounds=True)
        # Customer buys at 100.05: cash +100.045, inventory -1.
        # Cover at public 101.00 plus 0.025 spread and 0.005 fee: cash -101.03.
        self.assertAlmostEqual(stat["net_pnl"], -0.985, places=8)
        self.assertEqual(stat["hedge_volume"], 1)
        self.assertAlmostEqual(stat["hedge_cost_total"], 0.03)
        self.assertEqual(stat["max_abs_inventory"], 1)
        self.assertEqual(rows[0]["inventory_after"], 0)
        self.assertAlmostEqual(rows[0]["equity_before_hedge"], -0.955)
        self.assertAlmostEqual(rows[0]["equity_after"], -0.985)

    def test_wider_hedge_spread_lowers_pnl_on_same_market(self):
        cfg = Config(rounds=1)
        event = Event(1, 100.0, "buy", 102.0, 101.0, True)
        policy = Policy("hedged", 0.05, hedge_threshold=0)
        low, _ = simulate([event], cfg, policy)
        high, _ = simulate([event], replace(cfg, hedge_half_spread=0.075), policy)
        self.assertAlmostEqual(low["net_pnl"] - high["net_pnl"], 0.05)

    def test_no_fill_means_no_hedge(self):
        cfg = Config(rounds=1)
        event = Event(1, 100.0, "buy", 100.0, 101.0, True)
        stat, _ = simulate([event], cfg, Policy("hedged", 0.05, hedge_threshold=0))
        self.assertEqual((stat["fills"], stat["hedge_volume"], stat["net_pnl"]), (0, 0, 0.0))

    def test_hard_inventory_cap_blocks_sixth_same_side_fill(self):
        cfg = Config(rounds=6, max_inventory=5)
        events = [Event(i, 100.0, "buy", 101.0, 100.0, False) for i in range(1, 7)]
        stat, _ = simulate(events, cfg, Policy("unhedged", 0.05))
        self.assertEqual(stat["fills"], 5)
        self.assertEqual(stat["ending_inventory"], -5)
        self.assertEqual(stat["max_abs_inventory"], 5)
        self.assertAlmostEqual(stat["liquidation_cost"], 5 * (0.05 + 0.005))

    def test_constraint_gate_requires_stressed_participation(self):
        summary = {
            "mean_max_drawdown": 0.2,
            "tail_loss_wilson_upper95": 0.01,
            "stressed_mean_max_drawdown": 0.5,
            "stressed_fill_rate": 0.19,
        }
        self.assertFalse(feasible(summary))
        summary["stressed_fill_rate"] = 0.20
        self.assertTrue(feasible(summary))

    def test_paired_interval_matches_each_scenario(self):
        a = [{"regime": "calm", "seed": 1, "net_pnl": 3.0},
             {"regime": "calm", "seed": 2, "net_pnl": 5.0}]
        b = [{"regime": "calm", "seed": 2, "net_pnl": 2.0},
             {"regime": "calm", "seed": 1, "net_pnl": 1.0}]
        result = paired_interval(a, b, "net_pnl")
        self.assertEqual(result["pairs"], 2)
        self.assertAlmostEqual(result["mean_diff"], 2.5)


if __name__ == "__main__":
    unittest.main()
