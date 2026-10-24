"""Stage 4: how the drawdown penalty changes the fitted quoting policy.

One coefficient of the training objective varies; position and quoting bounds,
the candidate grid and the market generator are untouched. Every family is
re-tuned per coefficient and evaluated on shared exogenous paths.
"""
from __future__ import annotations

from pathlib import Path
from statistics import mean, stdev
from math import sqrt

from simulator import Config, Policy, simulate
from experiment import TRAIN_SEEDS, TEST_SEEDS, candidates, markets_for, evaluate, write_csv

# 0.8 is the stage 3 value; 0.3 is the more tolerant alternative compared here.
# 0 and 1.5 bracket them and are not selected on any test set.
LAMBDAS = (0.0, 0.3, 0.8, 1.5)
INVENTORY_PENALTY = 0.04
CONFIRM_SEEDS = tuple(range(60000, 60080))  # disjoint from train and previous test


def score(stat: dict, drawdown_penalty: float) -> float:
    return stat['net_pnl'] - drawdown_penalty * stat['max_drawdown'] - INVENTORY_PENALTY * stat['avg_abs_inventory']


def fit_all_preferences(cfg: Config, markets: dict) -> tuple[dict[float, list[Policy]], list[dict]]:
    """Simulate each candidate once, then re-score for every penalty value."""
    paths = list(markets.values())
    policy_metrics: dict[str, list[tuple[Policy,list[dict]]]] = {}
    for group in ('A', 'B', 'C'):
        policy_metrics[group] = [(p, [simulate(path, cfg, p)[0] for path in paths]) for p in candidates(group)]
    selections = {}
    rows = []
    for lam in LAMBDAS:
        best = []
        for group in ('A', 'B', 'C'):
            results = [(mean(score(s, lam) for s in stats), p) for p, stats in policy_metrics[group]]
            best_score, policy = max(results, key=lambda x: x[0])
            best.append(policy)
            rows.append({
                'drawdown_penalty': lam, 'group': group, 'strategy': policy.name,
                'base_half_spread': policy.base_half_spread,
                'inventory_alpha': policy.inventory_alpha,
                'volatility_beta': policy.volatility_beta,
                'training_objective': best_score,
                'max_inventory': cfg.max_inventory,
                'fee_per_trade': cfg.fee_per_trade,
            })
        selections[lam] = best
    return selections, rows


def summarize(rec: list[dict], lam: float, dataset: str) -> list[dict]:
    out = []
    for group, name in [('A','A_fixed'),('B','B_inventory'),('C','C_vol_adaptive')]:
        s = [r for r in rec if r['strategy']==name]
        out.append({
            'dataset': dataset, 'drawdown_penalty': lam, 'strategy': name,
            'n_markets': len(s), 'avg_net_pnl': mean(x['net_pnl'] for x in s),
            'sd_net_pnl': stdev(x['net_pnl'] for x in s),
            'avg_max_drawdown': mean(x['max_drawdown'] for x in s),
            'avg_abs_inventory': mean(x['avg_abs_inventory'] for x in s),
            'avg_fill_rate': mean(x['fill_rate'] for x in s),
            'negative_pnl_rate': mean(float(x['net_pnl']<0) for x in s),
            'avg_new_objective': mean(score(x,0.3) for x in s),
        })
    return out


def preference_differences(records_lo: list[dict], records_hi: list[dict], dataset: str) -> list[dict]:
    """Market-paired differences of the tolerant penalty against the original."""
    lookup_lo={(r['regime'],r['seed'],r['strategy']):r for r in records_lo}
    lookup_hi={(r['regime'],r['seed'],r['strategy']):r for r in records_hi}
    assert lookup_lo.keys()==lookup_hi.keys()
    rows=[]
    for strategy in ('A_fixed','B_inventory','C_vol_adaptive'):
        keys=[k for k in lookup_lo if k[2]==strategy]
        for m in ('net_pnl','max_drawdown','avg_abs_inventory','fill_rate'):
            d=[lookup_lo[k][m]-lookup_hi[k][m] for k in keys]
            center=mean(d)
            margin=1.96*stdev(d)/sqrt(len(d)) if len(d)>1 else float('nan')
            rows.append({'dataset':dataset,'strategy':strategy,'metric':m,'n_paired':len(d),
                         'mean_low_minus_original':center,
                         'ci95_low_approx':center-margin,'ci95_high_approx':center+margin})
    return rows


def run():
    assert not (set(TRAIN_SEEDS)&set(TEST_SEEDS) or set(TRAIN_SEEDS)&set(CONFIRM_SEEDS) or set(TEST_SEEDS)&set(CONFIRM_SEEDS))
    cfg=Config()
    train=markets_for(cfg, TRAIN_SEEDS)
    selected,params=fit_all_preferences(cfg,train)
    oldtest=markets_for(cfg, TEST_SEEDS)
    exploratory=[]
    records={}
    for lam in LAMBDAS:
        records[lam]=evaluate(oldtest,cfg,selected[lam])
        exploratory.extend(summarize(records[lam],lam,'previous_stage3_test_360'))
    # The two headline penalties, on markets not used anywhere above.
    confirmation_markets=markets_for(cfg,CONFIRM_SEEDS)
    confirm_low=evaluate(confirmation_markets,cfg,selected[0.3])
    confirm_high=evaluate(confirmation_markets,cfg,selected[0.8])
    confirmation=summarize(confirm_low,0.3,'fresh_confirmation_240')+summarize(confirm_high,0.8,'fresh_confirmation_240')
    paired=preference_differences(confirm_low,confirm_high,'fresh_confirmation_240')
    out=Path(__file__).resolve().parent/'output'
    write_csv(out/'trained_policies_by_risk_preference.csv',params)
    write_csv(out/'exploratory_sensitivity_360.csv',exploratory)
    write_csv(out/'fresh_confirmation_240.csv',confirmation)
    write_csv(out/'fresh_confirmation_paired_differences.csv',paired)
    # Per-market rows are written out so the summary tables can be recomputed.
    write_csv(out/'fresh_confirmation_raw_lower.csv',confirm_low)
    write_csv(out/'fresh_confirmation_raw_original.csv',confirm_high)
    print('=== training optimal parameters ===')
    for row in params: print(f"λ={row['drawdown_penalty']:.1f} {row['strategy']}: spread={row['base_half_spread']:.3f}, alpha={row['inventory_alpha']:.3f}, beta={row['volatility_beta']:.1f}")
    print('\n=== confirmation set (240 markets per policy, same 3 regimes) ===')
    for row in confirmation: print(f"λ={row['drawdown_penalty']:.1f} {row['strategy']}: pnl={row['avg_net_pnl']:.4f}, sd={row['sd_net_pnl']:.4f}, maxDD={row['avg_max_drawdown']:.4f}, |inventory|={row['avg_abs_inventory']:.3f}, fill={row['avg_fill_rate']:.1%}, newJ={row['avg_new_objective']:.4f}")
    print('\n=== paired differences, tolerant minus original penalty ===')
    for row in paired: print(f"{row['strategy']} {row['metric']}: {row['mean_low_minus_original']:+.4f}, 95% approx [{row['ci95_low_approx']:+.4f}, {row['ci95_high_approx']:+.4f}]")
    return params,exploratory,confirmation,paired

if __name__=='__main__':
    run()
