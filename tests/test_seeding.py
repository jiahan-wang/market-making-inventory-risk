"""Stream-uniqueness checks for the scenario generator.

markets_for() maps (seed, regime) onto one PRNG stream, keyed by
seed * len(REGIMES) + regime_index. The key has to be injective. If two splits ever
shared a stream they would draw the same standardized price path, and every
held-out number built on them would be a restatement of training. The obvious
alternative, seed + 100000 * regime_index, fails exactly that test here: the split
ranges are spaced 100000 apart, so the offset walks one split straight into
another.
"""
from __future__ import annotations

import unittest

from test_stages import stage

REGIMES = ("calm", "normal", "stressed")

# role -> seed ranges, per runnable stage protocol
PROTOCOLS = {
    3: ("experiment", ("TRAIN_SEEDS", "TEST_SEEDS")),
    4: ("risk_preference_experiment", ("TRAIN_SEEDS", "TEST_SEEDS", "CONFIRM_SEEDS")),
    5: ("risk_budget_optimizer", ("TRAIN_SEEDS", "VALIDATION_SEEDS", "TEST_SEEDS")),
    6: ("stress_budget_optimizer", ("TRAIN_SEEDS", "VALIDATION_SEEDS", "TEST_SEEDS")),
    7: ("liquidity_budget_optimizer", ("TRAIN_SEEDS", "VALIDATION_SEEDS", "TEST_SEEDS")),
}


def stream_ids(seeds, regime_count=len(REGIMES)):
    return {seed * regime_count + index for seed in seeds for index in range(regime_count)}


class StreamUniquenessTests(unittest.TestCase):
    def test_the_stream_key_is_injective(self):
        seen = {}
        for stage_no, (module, attrs) in PROTOCOLS.items():
            with stage(stage_no, module) as mod:
                for attr in attrs:
                    for seed in getattr(mod, attr):
                        for index in range(len(REGIMES)):
                            key = seed * len(REGIMES) + index
                            first = seen.setdefault(key, (stage_no, attr, seed))
                            same_owner = first[0] == stage_no and first[1] == attr
                            self.assertTrue(same_owner or first[2] == seed,
                                            f"stream {key} reused by {first} and {(stage_no, attr, seed)}")

    def test_no_split_within_a_protocol_shares_a_stream(self):
        for stage_no, (module, attrs) in PROTOCOLS.items():
            with stage(stage_no, module) as mod:
                ids = {attr: stream_ids(getattr(mod, attr)) for attr in attrs}
                for i, a in enumerate(attrs):
                    for b in attrs[i + 1:]:
                        self.assertFalse(ids[a] & ids[b],
                                         f"stage {stage_no}: {a} and {b} share streams")

    def test_the_stage7_final_split_is_disjoint_from_selection(self):
        with stage(7, "liquidity_budget_optimizer") as lbo:
            final = stream_ids(lbo.TEST_SEEDS)
            self.assertFalse(final & stream_ids(lbo.TRAIN_SEEDS))
            self.assertFalse(final & stream_ids(lbo.VALIDATION_SEEDS))

    def test_markets_for_produces_one_market_per_regime_and_seed(self):
        with stage(7, "simulator") as sim, stage(7, "experiment") as exp:
            cfg = sim.Config(rounds=3)
            markets = exp.markets_for(cfg, (1, 2, 3))
            self.assertEqual(len(markets), 3 * len(REGIMES))
            self.assertEqual({reg for reg, _ in markets}, set(REGIMES))
            paths = {(reg, seed): tuple((e.mid_before, e.mid_after) for e in events)
                     for (reg, seed), events in markets.items()}
            self.assertEqual(len(set(paths.values())), len(paths))

    def test_an_offset_scheme_would_alias_the_split_ranges(self):
        # The offset key is not injective: 100000-apart split ranges against a
        # 100000-sized regime offset collide, which is why the key multiplies.
        train = range(110000, 110120)
        validation = range(210000, 210080)
        old = lambda seeds, index: {s + 100000 * (index + 1) for s in seeds}
        collided = old(train, 1) & old(validation, 0)
        self.assertTrue(collided, "the old scheme is expected to alias")
        self.assertFalse(stream_ids(train) & stream_ids(validation))


if __name__ == "__main__":
    unittest.main()
