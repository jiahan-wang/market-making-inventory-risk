"""Risk-budget selection helpers for stage 6. Imported by stress_budget_optimizer.py.

Copied from stage5/risk_budget_optimizer.py rather than shared: it has to resolve
against this directory's simulator.py. Run stress_budget_optimizer.py, not this file.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from statistics import mean, stdev

from experiment import candidates, markets_for, write_csv
from simulator import Config, Policy, REGIMES, simulate

REGIME_NAMES = tuple(REGIMES)
TRAIN_SEEDS = tuple(range(110000, 110120))  # 360 markets, 120 per regime
VALIDATION_SEEDS = tuple(range(210000, 210080))  # 240 markets, 80 per regime
TEST_SEEDS = tuple(range(310000, 310120))  # 360 markets, 120 per regime

# Stated screening thresholds, chosen before the runs below.
MAX_MEAN_DRAWDOWN = 1.0
TAIL_LOSS_LEVEL = -2.0
MAX_TAIL_PROBABILITY = 0.05
Z_ONE_SIDED_95 = 1.6448536269514722


def wilson_upper(k: int, n: int, z: float = Z_ONE_SIDED_95) -> float:
    """One-sided 95% Wilson upper bound for Bernoulli event rate."""
    if not (n > 0 and 0 <= k <= n):
        raise ValueError('require 0 <= k <= n with n >= 1')
    p = k / n
    denom = 1 + z*z/n
    center = p + z*z/(2*n)
    spread = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return min(1.0, (center + spread)/denom)


def summarize_records(rows: list[dict]) -> dict:
    """Empirical risk criteria over an equal-weight mixture of the three regimes.

Markets are independent within a regime; the pooled Wilson bound is a screen
on this generator, not a tail-risk guarantee.
"""
    if not rows:
        raise ValueError('Need at least one market')
    n = len(rows)
    pnls = [x['net_pnl'] for x in rows]
    drawdowns = [x['max_drawdown'] for x in rows]
    bad = sum(x < TAIL_LOSS_LEVEL for x in pnls)
    return {
        'markets': n,
        'mean_net_pnl': mean(pnls),
        'sd_net_pnl': stdev(pnls) if n > 1 else 0.,
        'mean_max_drawdown': mean(drawdowns),
        'mean_abs_inventory': mean(x['avg_abs_inventory'] for x in rows),
        'mean_fill_rate': mean(x['fill_rate'] for x in rows),
        'max_abs_inventory_observed': max(x['max_abs_inventory'] for x in rows),
        'tail_loss_count': bad,
        'tail_loss_rate': bad/n,
        'tail_loss_wilson_upper95': wilson_upper(bad, n),
        'pnl_5pct': sorted(pnls)[math.ceil(.05*n)-1],
    }


def is_feasible(stat: dict) -> bool:
    """Two screens: average drawdown and the tail rate.

    The absolute-inventory cap used to be a third condition here. It could never
    fail: simulate() already refuses a fill that would push |inventory| past
    max_inventory, so every candidate reports max_abs_inventory_observed <= cap by
    construction. A screen that cannot fail is not a screen, so it is reported
    and not gated.
    """
    return (stat['mean_max_drawdown'] <= MAX_MEAN_DRAWDOWN
            and stat['tail_loss_wilson_upper95'] <= MAX_TAIL_PROBABILITY)


def bootstrap_mean_interval(values: list[float], reps: int = 2000,
                            level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap for the mean of a sample."""
    if len(values) < 2:
        return float('nan'), float('nan')
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(reps))
    return means[int((1 - level) / 2 * reps)], means[int((1 + level) / 2 * reps) - 1]


def bootstrap_pass_probability(values: list[float], threshold: float,
                               reps: int = 2000, seed: int = 0) -> float:
    """Share of bootstrap resamples whose mean clears the threshold.

    Two candidates can both print "met" while one sits half a standard error from
    the line and the other sits three. This separates them.
    """
    if not values:
        return float('nan')
    rng = random.Random(seed)
    n = len(values)
    hits = 0
    for _ in range(reps):
        if sum(values[rng.randrange(n)] for _ in range(n)) / n <= threshold:
            hits += 1
    return hits / reps


def evaluate_policy(policy: Policy, cfg: Config, market_items: list) -> list[dict]:
    out = []
    for (regime, seed), path in market_items:
        values, _ = simulate(path, cfg, policy)
        out.append({'regime': regime, 'seed': seed, **values})
    return out


def describe_policy(p: Policy) -> dict:
    return {'strategy': p.name, 'base_half_spread': p.base_half_spread,
            'inventory_alpha': p.inventory_alpha,
            'volatility_beta': p.volatility_beta}


def ranked_candidates(cfg: Config, train_items: list) -> tuple[list[tuple[Policy,dict]], list[dict]]:
    ranking = []
    detail = []
    for g in ('A','B','C'):
        for policy in candidates(g):
            stat = summarize_records(evaluate_policy(policy,cfg,train_items))
            stat['feasible_train'] = is_feasible(stat)
            ranking.append((policy, stat))
            detail.append({'split': 'train', **describe_policy(policy), **stat})
    return sorted(ranking, key=lambda t: t[1]['mean_net_pnl'], reverse=True), detail


def select_policy(cfg: Config, ranked: list[tuple[Policy,dict]], validation_items: list):
    """Walk policies in descending training mean P&L and stop at the first pass.

Validation is part of selection, so only the final set is untouched evidence.
"""
    validation = []
    selected = None
    for policy, train_stat in ranked:
        if not train_stat['feasible_train']:
            continue
        rows = evaluate_policy(policy, cfg, validation_items)
        stat = summarize_records(rows)
        stat['feasible_validation'] = is_feasible(stat)
        stat['dd_pass_probability'] = bootstrap_pass_probability(
            [r['max_drawdown'] for r in rows], MAX_MEAN_DRAWDOWN)
        validation.append({'split':'validation',**describe_policy(policy), **stat})
        if stat['feasible_validation']:
            selected = policy
            break
    return selected,validation


def paired_comparison(new: list[dict], base: list[dict], newer_name: str, older_name: str) -> list[dict]:
    lookup_new = {(r['regime'], r['seed']):r for r in new}
    lookup_base = {(r['regime'], r['seed']):r for r in base}
    if lookup_new.keys() != lookup_base.keys():
        raise ValueError('Cannot compare unmatched market scenarios')
    n = len(lookup_new)
    out = []
    for metric in ('net_pnl','max_drawdown','avg_abs_inventory','fill_rate'):
        diffs = [lookup_new[k][metric]-lookup_base[k][metric] for k in lookup_new]
        delta = mean(diffs)
        margin = 1.96*stdev(diffs)/math.sqrt(n) if n>1 else float('nan')
        out.append({'comparison':f'{newer_name} minus {older_name}', 'metric':metric,
                    'paired_markets':n, 'mean_difference':delta,
                    'approx_95_low':delta-margin,'approx_95_high':delta+margin})
    return out


def run(out_dir: Path | None = None):
    cfg = Config()
    assert cfg.max_inventory == 5
    assert not (set(TRAIN_SEEDS) & set(VALIDATION_SEEDS))
    assert not (set(TRAIN_SEEDS) & set(TEST_SEEDS))
    assert not (set(VALIDATION_SEEDS) & set(TEST_SEEDS))
    out_dir = out_dir or (Path(__file__).resolve().parent / 'output')
    out_dir.mkdir(parents=True,exist_ok=True)
    train_items = list(markets_for(cfg, TRAIN_SEEDS).items())
    validation_items = list(markets_for(cfg, VALIDATION_SEEDS).items())
    ranked, train_table = ranked_candidates(cfg,train_items)
    winner, val_table = select_policy(cfg,ranked,validation_items)
    write_csv(out_dir/'training_candidate_grid.csv',train_table)
    if val_table:
        write_csv(out_dir/'validation_screen.csv',val_table)
    else:
        (out_dir/'validation_screen.csv').write_text('No training-feasible candidates.\n', encoding='utf-8', newline='\n')
    if winner is None:
        report = 'No feasible policy: no candidate met both risk constraints on train and validation.\n'
        (out_dir/'results.txt').write_text(report, encoding='utf-8', newline='\n')
        print(report)
        return None

    # Reference policies are fixed here, not chosen on the final set:
    unconstrained = ranked[0][0]
    a = Policy('A_fixed',0.09)
    stage4_original = Policy('C_vol_adaptive', 0.09, 0.024, 0.4)
    # Distinct labels keep identically-parameterised policies from colliding.
    policies = {'budget_selected':winner, 'train_pnl_max':unconstrained,
                'fixed_baseline':a, 'stage4_original':stage4_original}
    test_items = list(markets_for(cfg, TEST_SEEDS).items())
    test_rows=[]
    stats={}
    rows_by={}
    regime_rows=[]
    for label,p in policies.items():
        rows=evaluate_policy(p,cfg,test_items)
        rows_by[label]=rows
        stat=summarize_records(rows)
        stat['feasible_empirical_test'] = is_feasible(stat)
        stats[label]=stat
        test_rows.append({'label':label,**describe_policy(p),**stat})
        for rg in REGIME_NAMES:
            r_stat=summarize_records([r for r in rows if r['regime']==rg])
            regime_rows.append({'label':label,'regime':rg,**describe_policy(p), **r_stat})

    write_csv(out_dir/'test_strategy_summary.csv',test_rows)
    write_csv(out_dir/'test_by_market_regime.csv',regime_rows)
    comparisons=[]
    for label in ('train_pnl_max','fixed_baseline','stage4_original'):
        comparisons.extend(paired_comparison(rows_by['budget_selected'],rows_by[label],
                                             'budget_selected',label))
    write_csv(out_dir/'test_paired_comparisons.csv',comparisons)
    write_csv(out_dir/'test_raw_360_market_results.csv',[
        {'label':label,**r} for label,rr in rows_by.items() for r in rr])
    lines=[
        'Stage 5: risk-budget selection on model-generated markets',
        'Splits: 360 train / 240 validation / 360 final markets; 3 regimes equally weighted; 150 rounds each.',
        f'Risk budget: mean maxDD <= 1.0; Wilson upper bound P(netPnL<{TAIL_LOSS_LEVEL})<={MAX_TAIL_PROBABILITY}; max |inventory|<={cfg.max_inventory}.',
        'Pooled Wilson bound over a stratified mixture; treat it as a screen, not a guarantee.',
        f'Train-feasible candidates: {sum(s["feasible_train"] for _, s in ranked)} / {len(ranked)}.',
        f'Validation candidates screened until one passed: {len(val_table)}.',
        f'SELECTED: {winner}',
        f'Highest training P&L, feasibility not checked: {unconstrained}',
        '', 'Final split (360 markets):',
    ]
    for r in test_rows:
        lines.append(f"{r['label']}: PnL={r['mean_net_pnl']:.4f}, SD={r['sd_net_pnl']:.4f}, avgDD={r['mean_max_drawdown']:.4f}, tail<-2={r['tail_loss_count']}/{r['markets']} ({r['tail_loss_rate']:.2%}), WilsonUpper95={r['tail_loss_wilson_upper95']:.2%}, avg|q|={r['mean_abs_inventory']:.3f}, fill={r['mean_fill_rate']:.2%}, test risk screen={r['feasible_empirical_test']}")
    lines.append('\nCOMPARISONS (BUDGET SELECTED - BENCHMARK):')
    for r in comparisons:
        if r['metric'] in ('net_pnl','max_drawdown'):
            lines.append(f"{r['comparison']} {r['metric']}: {r['mean_difference']:+.4f}; approx 95% [{r['approx_95_low']:+.4f}, {r['approx_95_high']:+.4f}]")
    (out_dir/'results.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    print('\n'.join(lines))
    return {'winner':winner,'test':test_rows,'regimes':regime_rows,'comparisons':comparisons,'ranking':train_table,'validation':val_table}


