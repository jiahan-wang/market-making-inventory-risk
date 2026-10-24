"""Mechanics checks for stages 1 to 6.

Each stage directory carries its own copy of simulator.py and experiment.py, so a
module is imported with that directory first on sys.path and the previous stage is
evicted around it.
"""
from __future__ import annotations

import contextlib
import importlib
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODULES = ("simulator", "experiment", "risk_budget_optimizer",
                 "stress_budget_optimizer", "liquidity_budget_optimizer",
                 "risk_preference_experiment")


@contextlib.contextmanager
def stage(stage_no, name):
    saved_path = list(sys.path)
    saved = {k: sys.modules[k] for k in LOCAL_MODULES if k in sys.modules}
    for key in saved:
        sys.modules.pop(key)
    sys.path.insert(0, str(ROOT / f"stage{stage_no}"))
    try:
        yield importlib.import_module(name)
    finally:
        sys.path[:] = saved_path
        for key in LOCAL_MODULES:
            sys.modules.pop(key, None)
        sys.modules.update(saved)


class Stage1Tests(unittest.TestCase):
    def test_cap_cash_and_equity_hold_on_every_round(self):
        with stage(1, "simulator") as sim:
            rows, trader, stats = sim.simulate(sim.Config(n_rounds=200, max_inventory=3))
            self.assertTrue(all(abs(r["inventory_after"]) <= 3 for r in rows))
            self.assertEqual(stats["ending_inventory"], trader.inventory)
            for r in rows:
                self.assertAlmostEqual(r["equity_after"],
                                       r["cash_after"] + r["inventory_after"] * r["mid_after"],
                                       places=3)

    def test_a_fill_needs_both_price_acceptance_and_room(self):
        with stage(1, "simulator") as sim:
            rows, _, _ = sim.simulate(sim.Config(n_rounds=300))
            for r in rows:
                self.assertEqual(r["filled"], int(r["price_acceptable"] and r["inventory_permits"]))

    def test_quotes_are_symmetric_around_the_mid(self):
        with stage(1, "simulator") as sim:
            cfg = sim.Config(half_spread=0.05)
            bid, ask = sim.make_quotes(100.0, cfg)
            self.assertAlmostEqual(ask - bid, 2 * cfg.half_spread)
            self.assertAlmostEqual((bid + ask) / 2, 100.0)

    def test_the_same_seed_replays_the_same_path(self):
        with stage(1, "simulator") as sim:
            a, _, _ = sim.simulate(sim.Config(n_rounds=20))
            b, _, _ = sim.simulate(sim.Config(n_rounds=20))
            self.assertEqual([r["mid_after"] for r in a], [r["mid_after"] for r in b])


class Stage2Tests(unittest.TestCase):
    def test_both_strategies_see_one_identical_market(self):
        with stage(2, "simulator") as sim:
            rows, summaries = sim.compare_once(sim.Config(n_rounds=40))
            paths: dict = {}
            for r in rows:
                paths.setdefault(r["strategy"], []).append(
                    (r["round"], r["mid_before"], r["mid_after"]))
            self.assertEqual(len(paths), 2)
            self.assertEqual(len({tuple(sorted(v)) for v in paths.values()}), 1)
            self.assertEqual({s["strategy"] for s in summaries}, set(paths))

    def test_inventory_skew_moves_quotes_away_from_the_position(self):
        with stage(2, "simulator") as sim:
            cfg = sim.Config(half_spread=0.05, inventory_alpha=0.01)
            flat_bid, flat_ask = sim.make_quotes(100.0, 0, "inventory_aware", cfg)
            long_bid, long_ask = sim.make_quotes(100.0, 4, "inventory_aware", cfg)
            self.assertLess(long_ask, flat_ask)
            self.assertLess(long_bid, flat_bid)

    def test_liquidation_is_charged_on_the_remaining_position(self):
        with stage(2, "simulator") as sim:
            cfg = sim.Config(n_rounds=60, liquidation_half_spread=0.05, fee_per_trade=0.005)
            _, stats = sim.simulate(sim.generate_market(cfg), cfg, "fixed")
            self.assertEqual(stats["liquidation_cost"],
                             abs(stats["ending_inventory"]) * (cfg.liquidation_half_spread + cfg.fee_per_trade))
            self.assertAlmostEqual(stats["ending_equity_marked"] - stats["liquidation_cost"],
                                   stats["ending_pnl_after_liquidation"], places=9)

    def test_repeated_comparison_is_seed_deterministic(self):
        with stage(2, "simulator") as sim:
            key = lambda rec: (rec["seed"], rec["strategy"], rec["ending_pnl_after_liquidation"])
            a, _ = sim.repeated_comparison(sim.Config(n_rounds=10), n_scenarios=4, seed_start=777)
            b, _ = sim.repeated_comparison(sim.Config(n_rounds=10), n_scenarios=4, seed_start=777)
            self.assertEqual([key(r) for r in a], [key(r) for r in b])
            self.assertEqual(len({r["seed"] for r in a}), 4)


class Stage3Tests(unittest.TestCase):
    def test_quote_is_invariant_to_the_arriving_order_and_the_next_mid(self):
        with stage(3, "simulator") as sim:
            cfg = sim.Config(rounds=3)
            policy = sim.Policy("p", 0.05)
            up = sim.Event(1, 100.0, "buy", 103.0, 102.0, True)
            down = sim.Event(1, 100.0, "sell", 97.0, 98.0, False)
            self.assertEqual(sim.make_quotes(up.mid_before, 0, cfg.vol_reference, policy, cfg),
                             sim.make_quotes(down.mid_before, 0, cfg.vol_reference, policy, cfg))

    def test_informed_flow_marks_out_worse_than_noise(self):
        with stage(3, "simulator") as sim:
            cfg, policy = sim.Config(), sim.Policy("p", 0.05)
            inf, noi = [], []
            for seed in range(40000, 40060):
                for regime in sim.REGIMES:
                    stats, _ = sim.simulate(sim.build_market(cfg, regime, seed), cfg, policy)
                    for key, bucket in (("mean_informed_markout", inf), ("mean_noise_markout", noi)):
                        value = stats[key]
                        if not math.isnan(value):
                            bucket.append(value)
            self.assertLess(sum(inf) / len(inf), sum(noi) / len(noi))

    def test_volatility_estimator_looks_back_only(self):
        with stage(3, "simulator") as sim:
            cfg = sim.Config(vol_window=2)
            self.assertAlmostEqual(sim.recent_volatility([0.3, 0.1, 0.1], cfg), 0.1)
            self.assertEqual(sim.recent_volatility([], cfg), cfg.vol_reference)

    def test_the_inventory_cap_blocks_the_sixth_same_side_fill(self):
        with stage(3, "simulator") as sim:
            cfg = sim.Config(rounds=6, max_inventory=5)
            events = [sim.Event(i, 100.0, "buy", 101.0, 100.0, False) for i in range(1, 7)]
            stats, _ = sim.simulate(events, cfg, sim.Policy("u", 0.05))
            self.assertEqual((stats["fills"], stats["ending_inventory"]), (5, -5))


class Stage4Tests(unittest.TestCase):
    def test_each_penalty_selects_its_own_family_best(self):
        with stage(4, "simulator") as sim, stage(4, "experiment") as exp, \
                stage(4, "risk_preference_experiment") as rpe:
            cfg = sim.Config()
            markets = exp.markets_for(cfg, exp.TRAIN_SEEDS)
            paths = list(markets.values())
            selections = rpe.fit_all_preferences(cfg, markets)[0]
            for lam in rpe.LAMBDAS:
                for index, group in enumerate(("A", "B", "C")):
                    scored = []
                    for cand in exp.candidates(group):
                        values = [rpe.score(sim.simulate(path, cfg, cand)[0], lam) for path in paths]
                        scored.append((sum(values) / len(values), cand))
                    best = max(scored, key=lambda t: t[0])[1]
                    chosen = selections[lam][index]
                    self.assertEqual(
                        (best.name, best.base_half_spread, best.inventory_alpha, best.volatility_beta),
                        (chosen.name, chosen.base_half_spread, chosen.inventory_alpha, chosen.volatility_beta),
                        f"lambda={lam} group={group}")

    def test_preference_differences_pair_market_by_market(self):
        with stage(4, "simulator") as sim, stage(4, "experiment") as exp, \
                stage(4, "risk_preference_experiment") as rpe:
            cfg = sim.Config()
            markets = exp.markets_for(cfg, (60000, 60001))
            names = {"A_fixed": 0.09, "B_inventory": 0.07, "C_vol_adaptive": 0.05}
            lo = exp.evaluate(markets, cfg, [sim.Policy(n, s) for n, s in names.items()])
            hi = exp.evaluate(markets, cfg, [sim.Policy(n, s + 0.01) for n, s in names.items()])
            rows = rpe.preference_differences(lo, hi, "unit")
            self.assertEqual(len(rows), 3 * 4)
            self.assertTrue(all(r["n_paired"] == len(markets) for r in rows))
            self.assertTrue(all(r["mean_low_minus_original"] == 0.0
                                for r in rpe.preference_differences(lo, lo, "unit")))


class Stage5Tests(unittest.TestCase):
    def test_wilson_bound_inverts_the_score_statistic(self):
        with stage(5, "risk_budget_optimizer") as rbo:
            z = rbo.Z_ONE_SIDED_95
            for k, n in ((0, 100), (1, 100), (5, 100), (50, 100), (2, 360), (9, 360)):
                p_hat = k / n
                lo, hi = p_hat, 1.0
                for _ in range(200):
                    mid = (lo + hi) / 2
                    if n * (mid - p_hat) ** 2 - z * z * mid * (1 - mid) <= 0:
                        lo = mid
                    else:
                        hi = mid
                self.assertAlmostEqual(rbo.wilson_upper(k, n), (lo + hi) / 2, places=12)
            with self.assertRaises(ValueError):
                rbo.wilson_upper(101, 100)

    def test_the_gate_checks_drawdown_and_the_tail_rate(self):
        with stage(5, "risk_budget_optimizer") as rbo:
            ok = {"mean_max_drawdown": 0.5, "tail_loss_wilson_upper95": 0.02}
            self.assertTrue(rbo.is_feasible(ok))
            for field, bad in (("mean_max_drawdown", 1.5), ("tail_loss_wilson_upper95", 0.06)):
                self.assertFalse(rbo.is_feasible(dict(ok, **{field: bad})), field)

    def test_the_inventory_cap_comes_from_the_simulator(self):
        # The cap is a hard constraint inside simulate(), not a screen: the gate
        # cannot see a candidate that breaks it, so it must not pretend to test it.
        with stage(5, "simulator") as sim:
            cfg = sim.Config(rounds=40, max_inventory=3)
            events = sim.build_market(cfg, "stressed", 7)
            stats, _ = sim.simulate(events, cfg, sim.Policy("u", 0.005))
            self.assertEqual(stats["max_abs_inventory"], cfg.max_inventory)

    def test_summary_rejects_an_empty_market_set(self):
        with stage(5, "risk_budget_optimizer") as rbo:
            with self.assertRaises(ValueError):
                rbo.summarize_records([])

    def test_paired_comparison_refuses_unmatched_scenarios(self):
        with stage(5, "risk_budget_optimizer") as rbo:
            def row(seed):
                return {"regime": "calm", "seed": seed, "net_pnl": 1.0, "max_drawdown": 0.1,
                        "avg_abs_inventory": 1.0, "fill_rate": 0.5}
            with self.assertRaises(ValueError):
                rbo.paired_comparison([row(1)], [row(2)], "a", "b")


class Stage6Tests(unittest.TestCase):
    def test_stress_summary_requires_stressed_markets(self):
        with stage(6, "stress_budget_optimizer") as sbo:
            rows = [{"regime": "calm", "seed": 1, "net_pnl": 0.5, "max_drawdown": 0.1,
                     "avg_abs_inventory": 0.0, "fill_rate": 0.2, "max_abs_inventory": 1,
                     "hedge_volume": 0, "hedge_cost_total": 0.0}]
            with self.assertRaises(ValueError):
                sbo.summarize_with_stress(rows)

    def test_stressed_budget_is_a_separate_gate(self):
        with stage(6, "stress_budget_optimizer") as sbo:
            stat = {"mean_max_drawdown": 0.2, "tail_loss_wilson_upper95": 0.01,
                    "stressed_mean_max_drawdown": 1.4}
            self.assertFalse(sbo.within_all_budgets(stat))
            stat["stressed_mean_max_drawdown"] = 0.9
            self.assertTrue(sbo.within_all_budgets(stat))

    def test_terminal_liquidation_lands_on_the_equity_path(self):
        with stage(6, "simulator") as sim:
            cfg = sim.Config(rounds=2)
            events = [sim.Event(i, 100.0, "sell", 99.0, 100.0, False) for i in (1, 2)]
            stats, rows = sim.simulate(events, cfg, sim.Policy("u", 0.01), record_rounds=True)
            self.assertEqual(stats["ending_inventory"], 2)
            self.assertAlmostEqual(stats["liquidation_cost"],
                                   2 * (cfg.liquidation_half_spread + cfg.fee_per_trade))
            self.assertGreaterEqual(stats["max_drawdown"], stats["max_drawdown_before_liquidation"])
            self.assertEqual(rows[-1]["event_kind"], "terminal_liquidation")

    def test_a_mixed_row_table_still_writes(self):
        with stage(6, "experiment") as exp:
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "mixed.csv"
                exp.write_csv(target, [{"a": 1}, {"a": 2, "b": 3}])
                self.assertEqual(target.read_text(encoding="utf-8-sig").splitlines()[0], "a,b")


if __name__ == "__main__":
    unittest.main()
