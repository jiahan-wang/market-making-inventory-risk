"""Stage 6: adds a stressed-regime drawdown budget to the stage 5 screens.

The budgets were written down before the splits were drawn. Run this file; it
rewrites stage6/output.
"""
from __future__ import annotations

from pathlib import Path
from statistics import mean

from experiment import candidates, markets_for, write_csv
from simulator import Config, Policy, REGIMES
from risk_budget_optimizer import (
    describe_policy, evaluate_policy, summarize_records,
    is_feasible, paired_comparison, MAX_MEAN_DRAWDOWN,
    MAX_TAIL_PROBABILITY, TAIL_LOSS_LEVEL, bootstrap_pass_probability,
)

TRAIN_SEEDS = tuple(range(610000, 610120))
VALIDATION_SEEDS = tuple(range(710000, 710080))
TEST_SEEDS = tuple(range(810000, 810120))
STRESSED_MAX_MEAN_DRAWDOWN = 1.0


def summarize_with_stress(rows: list[dict]) -> dict:
    stat = summarize_records(rows)
    stressed = [row for row in rows if row['regime'] == 'stressed']
    if not stressed:
        raise ValueError('Missing stressed markets for regime risk screen')
    stat['stressed_markets'] = len(stressed)
    stat['stressed_mean_max_drawdown'] = mean(row['max_drawdown'] for row in stressed)
    stat['stressed_mean_net_pnl'] = mean(row['net_pnl'] for row in stressed)
    stat['stressed_tail_count'] = sum(row['net_pnl'] < TAIL_LOSS_LEVEL for row in stressed)
    return stat


def within_all_budgets(stat: dict) -> bool:
    return (is_feasible(stat)
            and stat['stressed_mean_max_drawdown'] <= STRESSED_MAX_MEAN_DRAWDOWN)


def score_candidates(cfg: Config, items: list) -> list[tuple[Policy,dict]]:
    scored = []
    for group in ('A', 'B', 'C'):
        for p in candidates(group):
            summary = summarize_with_stress(evaluate_policy(p, cfg, items))
            summary['feasible'] = within_all_budgets(summary)
            scored.append((p, summary))
    return sorted(scored, key=lambda item:item[1]['mean_net_pnl'], reverse=True)


def select_validated(cfg: Config, train_ranked: list[tuple[Policy,dict]], validation_items: list):
    """Validation feeds selection, so it is not an unbiased performance estimate."""
    checks = []
    for p, stat in train_ranked:
        if not stat['feasible']:
            continue
        rows = evaluate_policy(p, cfg, validation_items)
        val = summarize_with_stress(rows)
        val['feasible'] = within_all_budgets(val)
        val['stressed_dd_pass_probability'] = bootstrap_pass_probability(
            [r['max_drawdown'] for r in rows if r['regime'] == 'stressed'],
            STRESSED_MAX_MEAN_DRAWDOWN)
        checks.append((p,val))
        if val['feasible']:
            return p,checks
    return None,checks


def run(out_dir: Path | None=None):
    cfg = Config()
    assert len(set(TRAIN_SEEDS) & set(VALIDATION_SEEDS))==0
    assert len(set(TRAIN_SEEDS) & set(TEST_SEEDS))==0
    assert len(set(VALIDATION_SEEDS) & set(TEST_SEEDS))==0
    out_dir = out_dir or Path(__file__).resolve().parent / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)

    train_items = list(markets_for(cfg, TRAIN_SEEDS).items())
    val_items = list(markets_for(cfg, VALIDATION_SEEDS).items())
    ranked = score_candidates(cfg, train_items)
    winner, checked = select_validated(cfg, ranked, val_items)
    write_csv(out_dir/'stage6_training_candidates.csv',
              [{'split':'train',**describe_policy(p),**stat} for p,stat in ranked])
    if checked:
        write_csv(out_dir/'stage6_validation_screen.csv',
                  [{'split':'validation',**describe_policy(p),**stat} for p,stat in checked])
    else:
        (out_dir/'stage6_validation_screen.csv').write_text('No training-feasible candidates checked.\n', encoding='utf-8', newline='\n')

    lines = [
        'Stage 6: stressed-regime drawdown budget, model-generated markets',
        'Terminal liquidation is part of both the drawdown path and net P&L.',
        '39 stated candidates; 360 train / 240 validation / 360 final markets.',
        f'Budget: overall mean DD <= {MAX_MEAN_DRAWDOWN}; stressed mean DD <= {STRESSED_MAX_MEAN_DRAWDOWN};',
        f'  Wilson upper 95% tail rate for net PnL < {TAIL_LOSS_LEVEL} <= {MAX_TAIL_PROBABILITY}.',
        f'  |q| <= {cfg.max_inventory} is enforced inside simulate(); reported, not gated.',
        f'Train feasible: {sum(stat["feasible"] for _,stat in ranked)} / {len(ranked)}.',
        f'Validation candidates checked: {len(checked)}.',
        f'Selected policy: {winner}.',
    ]
    if winner is None:
        lines.append('No feasible policy on train and validation; the final split was not read.')
        report='\n'.join(lines)+'\n'
        (out_dir/'results.txt').write_text(report, encoding='utf-8', newline='\n')
        print(report)
        return {'selected':None, 'ranking':ranked, 'validation':checked}

    # Reference policies are fixed here, not chosen on the final set:
    old_budget = Policy('C_vol_adaptive', 0.09, 0.016, 0.4)  # selected on Stage-5 prior splits
    fixed = Policy('A_fixed', 0.09)
    train_pnl_highest = ranked[0][0]
    policies={
        'stress_budget_selected': winner,
        'stage5_budget_reference': old_budget,
        'training_profit_max': train_pnl_highest,
        'fixed_spread_reference': fixed,
    }
    test_items = list(markets_for(cfg, TEST_SEEDS).items())
    summaries=[]
    per_regime=[]
    raw=[]
    cached={}
    for label, p in policies.items():
        data = evaluate_policy(p, cfg, test_items)
        cached[label]=data
        stat = summarize_with_stress(data)
        stat['meets_test_empirical_budgets'] = within_all_budgets(stat)
        summaries.append({'label':label,**describe_policy(p),**stat})
        for reg in REGIMES:
            reg_data=[r for r in data if r['regime']==reg]
            per_regime.append({'label':label, 'regime':reg,**describe_policy(p),**summarize_records(reg_data)})
        raw.extend({'label':label,**record} for record in data)
    write_csv(out_dir/'test_360_strategy_summary.csv',summaries)
    write_csv(out_dir/'test_regime_summary.csv',per_regime)
    write_csv(out_dir/'test_360_raw.csv',raw)
    comparisons=[]
    for label in list(policies)[1:]:
        comparisons.extend(paired_comparison(cached['stress_budget_selected'],cached[label],
                                            'stress_budget_selected',label))
    write_csv(out_dir/'test_paired_comparisons.csv',comparisons)
    for row in summaries:
        lines.append(f"{row['label']}: meanPnL={row['mean_net_pnl']:.4f}, meanDD={row['mean_max_drawdown']:.4f}, "
                     f"stressMeanDD={row['stressed_mean_max_drawdown']:.4f}, |q|={row['mean_abs_inventory']:.4f}, "
                     f"tail={row['tail_loss_count']}/{row['markets']}, WilsonUpper={row['tail_loss_wilson_upper95']:.4%}, "
                     f"riskBudgetMetOnTest={row['meets_test_empirical_budgets']}")
    for r in comparisons:
        if r['metric'] in ('net_pnl','max_drawdown'):
            lines.append(f"{r['comparison']} / {r['metric']}: {r['mean_difference']:+.4f}; "
                         f"approx95=[{r['approx_95_low']:+.4f}, {r['approx_95_high']:+.4f}]")
    lines.append('Screening uses empirical mean drawdown and a pooled Wilson bound over a stratified mixture.')
    report='\n'.join(lines)+'\n'
    (out_dir/'results.txt').write_text(report, encoding='utf-8', newline='\n')
    print(report)
    return {'selected':winner,'ranking':ranked,'validation':checked, 'summary':summaries,
            'per_regime':per_regime,'comparisons':comparisons}


if __name__=='__main__':
    run()
