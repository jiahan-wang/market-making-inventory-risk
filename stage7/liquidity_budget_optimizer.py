"""Stage 7: quoting plus paid external hedging under a stressed-participation floor.

The hedged half of the candidate grid was drafted from a 30-market stressed run on
the stage 6 generator. That run plays no part in the selection below; train,
validation and final sets all come from their own seed ranges.
"""
from __future__ import annotations
from dataclasses import replace
from itertools import product
from pathlib import Path
from statistics import mean, stdev
from math import sqrt
from experiment import candidates, markets_for, write_csv
from simulator import Config, Policy, REGIMES, simulate
from risk_budget_optimizer import (evaluate_policy, summarize_records, describe_policy,
                                   bootstrap_mean_interval, bootstrap_pass_probability)
from stress_budget_optimizer import (summarize_with_stress, within_all_budgets,
                                     STRESSED_MAX_MEAN_DRAWDOWN)
# Disjoint from every earlier stage's seed ranges.
TRAIN_SEEDS = tuple(range(920000, 920120))
VALIDATION_SEEDS = tuple(range(930000, 930080))
TEST_SEEDS = tuple(range(940000, 940120))
STRESSED_MIN_FILL_RATE = 0.20
def candidate_policies() -> list[Policy]:
    """39 unhedged candidates plus 20 hedged ones.
At threshold 0 inventory is flat at every quote, so the skew coefficient is
inert and is fixed rather than swept; the 1/2-threshold arms do sweep it.
"""
    base = [p for name in 'ABC' for p in candidates(name)]
    extended=[]
    for d,b in product((0.07,0.09),(0.4,1.0)):
        extended.append(Policy('D_external_hedge',d,0.016,b,0))
        for alpha,threshold in product((0.016,0.024),(1,2)):
            extended.append(Policy('D_external_hedge',d,alpha,b,threshold))
    assert len(base)==39 and len(extended)==20
    return base+extended
def liquidity_summary(rows:list[dict]) -> dict:
    stats = summarize_with_stress(rows)
    sr=[r for r in rows if r['regime']=='stressed']
    stats['stressed_fill_rate']=mean(r['fill_rate'] for r in sr)
    stats['stressed_mean_hedge_volume']=mean(r['hedge_volume'] for r in sr)
    stats['stressed_mean_hedge_cost']=mean(r['hedge_cost_total'] for r in sr)
    stats['mean_hedge_volume']=mean(r['hedge_volume'] for r in rows)
    stats['mean_hedge_cost']=mean(r['hedge_cost_total'] for r in rows)
    return stats
def feasible(stats:dict) -> bool:
    return (within_all_budgets(stats)
            and stats['stressed_fill_rate']>=STRESSED_MIN_FILL_RATE)
def score_all(cfg:Config, train_items:list) -> list[tuple[Policy,dict]]:
    ranked=[]
    for p in candidate_policies():
        stat=liquidity_summary(evaluate_policy(p,cfg,train_items))
        stat['feasible']=feasible(stat)
        ranked.append((p,stat))
    return sorted(ranked, key=lambda t:t[1]['mean_net_pnl'], reverse=True)
def select(cfg:Config, ranked:list[tuple[Policy,dict]], validation_items:list):
    validations=[]
    for policy, train_stats in ranked:
        if not train_stats['feasible']:
            continue
        rows=evaluate_policy(policy,cfg,validation_items)
        val=liquidity_summary(rows)
        val['feasible']=feasible(val)
        # How much of the gate is signal: the share of bootstrap resamples of the
        # 80 stressed markets whose mean still clears the drawdown budget.
        val['stressed_dd_pass_probability']=bootstrap_pass_probability(
            [r['max_drawdown'] for r in rows if r['regime']=='stressed'],
            STRESSED_MAX_MEAN_DRAWDOWN)
        validations.append((policy,val))
        if val['feasible']:
            return policy,validations
    return None,validations
def describe(p: Policy) -> dict:
    row = {**describe_policy(p),
           'hedge_threshold': p.hedge_threshold if p.hedge_threshold is not None else 'none'}
    # At threshold 0 inventory is flat whenever a quote is formed, so the skew
    # coefficient cannot change behaviour. Flag it rather than editing the value.
    row['inventory_alpha_inert'] = p.hedge_threshold == 0
    return row
def records_by_regime(rows:list[dict]):
    out={}
    for reg in REGIMES:
        subset=[r for r in rows if r['regime']==reg]
        stats=summarize_records(subset)
        stats['mean_hedge_volume']=mean(r['hedge_volume'] for r in subset)
        stats['mean_hedge_cost']=mean(r['hedge_cost_total'] for r in subset)
        out[reg]=stats
    return out
def paired_interval(a:list[dict], b:list[dict], key:str)->dict:
    # Matches by market; the same price shocks, order sides, customer values
    # and hidden signals are used for both strategies.
    by_b={(x['regime'],x['seed']):x for x in b}
    values=[r[key]-by_b[(r['regime'],r['seed'])][key] for r in a]
    avg=mean(values)
    sem=stdev(values)/sqrt(len(values))
    return {'metric':key,'pairs':len(values),'mean_diff':avg,
            'approx_95_lower':avg-1.96*sem,'approx_95_upper':avg+1.96*sem}
def run(out_dir:Path|None=None)->dict:
    cfg=Config()
    assert set(TRAIN_SEEDS).isdisjoint(VALIDATION_SEEDS)
    assert set(TRAIN_SEEDS).isdisjoint(TEST_SEEDS)
    assert set(VALIDATION_SEEDS).isdisjoint(TEST_SEEDS)
    out_dir=out_dir or Path(__file__).resolve().parent/'output'
    out_dir.mkdir(parents=True,exist_ok=True)
    train=list(markets_for(cfg,TRAIN_SEEDS).items())
    val=list(markets_for(cfg,VALIDATION_SEEDS).items())
    ranking=score_all(cfg,train)
    write_csv(out_dir/'training_59_candidates.csv',
              [{'split':'train',**describe(p),**s} for p,s in ranking])
    selected,checks=select(cfg,ranking,val)
    if checks:
        write_csv(out_dir/'validation_screen.csv',
                  [{'split':'validation',**describe(p),**s} for p,s in checks])
    else:
        (out_dir/'validation_screen.csv').write_text('No feasible training candidates; validation not used.\n', encoding='utf8', newline='\n')
    old39=[(p,s) for p,s in ranking if p.hedge_threshold is None]
    lines=[
        'Stage 7: stressed participation floor and paid external hedging',
        'Candidate grid: 39 unhedged plus 20 hedged configurations.',
        'Grid drafting used a 30-market stressed exercise on the stage 6 generator; selection below does not.',
        'Splits: 360 train, 240 validation, 360 final markets, 150 rounds each.',
        'Screens: stressed mean maxDD<=1; all-regime mean maxDD<=1; Wilson upper tail-event rate<=5%;',
        f'  stressed mean fill rate >= {STRESSED_MIN_FILL_RATE:.0%} (a participation floor, not an exchange obligation).',
        f'  |q|<={cfg.max_inventory} is enforced inside simulate(); reported, not gated.',
        f'Unhedged candidates meeting the stressed risk and participation screens in train: {sum(feasible(s) for p,s in old39)}/39.',
        f'Extended grid train feasible: {sum(s["feasible"] for p,s in ranking if p.hedge_threshold is not None)}/20.',
        f'All-grid train feasible: {sum(s["feasible"] for _,s in ranking)}/{len(ranking)}.',
        f'Validated candidates checked: {len(checks)}; selected: {selected}.',
        'Hedges are struck at the mid that becomes public after the fill, plus a fixed half-spread and fee;',
        '  assumed infinite depth and immediate execution: no venue impact, latency or regime-dependent spread.',
    ]
    if selected is None:
        lines.append('No validated feasible policy; the final split was not read.')
        report='\n'.join(lines)+'\n'
        (out_dir/'results.txt').write_text(report, encoding='utf8', newline='\n')
        print(report)
        return {'selected':None,'ranking':ranking,'validation':checks}
    # Reference policies are fixed here. There is no external preregistration
    # record for them.
    refs={
        'stage7_selected':selected,
        'stage6_risk_only':Policy('C_vol_adaptive',.09,.024,1.6),
        'stage5_reference':Policy('C_vol_adaptive',.09,.016,.4),
    }
    test=list(markets_for(cfg,TEST_SEEDS).items())  # read once, then reused for sensitivity
    raw=[]; table=[]; regime_table=[]; cache={}
    for label,p in refs.items():
        data=evaluate_policy(p,cfg,test)
        cache[label]=data
        stat=liquidity_summary(data)
        stat['meets_test_empirical_constraints']=feasible(stat)
        table.append({'label':label,**describe(p),**stat})
        raw.extend({'label':label,**r} for r in data)
        for reg,values in records_by_regime(data).items():
            regime_table.append({'label':label,'regime':reg,**describe(p),**values})
    write_csv(out_dir/'test_360_policy_summary.csv',table)
    write_csv(out_dir/'test_regimes.csv',regime_table)
    write_csv(out_dir/'test_360_raw.csv',raw)
    comparisons=[]
    for label in list(refs)[1:]:
        for metric in ('net_pnl','max_drawdown','avg_abs_inventory','fill_rate','hedge_cost_total'):
            comparisons.append({'reference':label,**paired_interval(cache['stage7_selected'],cache[label],metric)})
    write_csv(out_dir/'test_paired_approx95.csv',comparisons)
    # Same markets, same policy, only the venue cost changes: a counterfactual on
    # the final set, not a further independent sample.
    sens=[]
    for half_spread in (cfg.hedge_half_spread, 0.05,0.075,0.10):
        new_cfg=replace(cfg,hedge_half_spread=half_spread)
        data=evaluate_policy(selected,new_cfg,test)
        stat=liquidity_summary(data)
        low,high=bootstrap_mean_interval(
            [r['max_drawdown'] for r in data if r['regime']=='stressed'])
        sens.append({'external_hedge_half_spread':half_spread,
                     'stressed_dd_ci_low':low,'stressed_dd_ci_high':high,
                     **stat,'meets_empirical_constraints':feasible(stat)})
    write_csv(out_dir/'hedge_cost_sensitivity.csv',sens)
    # Round-level ledger for one stressed market, so the accounting can be inspected
    # row by row rather than inferred from the summary tables.
    # The most active stressed market, so the ledger shows fills, hedges and
    # adverse moves rather than an empty round sequence.
    stressed = [items for items in test if items[0][0] == 'stressed']
    ledger_market = max(stressed, key=lambda items: simulate(items[1], cfg, selected)[0]['fills'])
    _, ledger_rows = simulate(ledger_market[1], cfg, selected, record_rounds=True)
    write_csv(out_dir/'selected_market_ledger.csv', ledger_rows)
    for row in table:
        lines.append(f"{row['label']}: pnl={row['mean_net_pnl']:.4f}; meanMaxDD={row['mean_max_drawdown']:.4f}; stressMaxDD={row['stressed_mean_max_drawdown']:.4f}; stressFill={row['stressed_fill_rate']:.2%}; hedgeCost={row['mean_hedge_cost']:.4f}; constraintScreen={row['meets_test_empirical_constraints']}")
    for row in sens:
        lines.append(f"HEDGE COST SENSITIVITY: hedgeHalfSpread={row['external_hedge_half_spread']:.3f}; pnl={row['mean_net_pnl']:.4f}; "
                     f"stressMaxDD={row['stressed_mean_max_drawdown']:.4f} "
                     f"[{row['stressed_dd_ci_low']:.4f},{row['stressed_dd_ci_high']:.4f}]; stressFill={row['stressed_fill_rate']:.2%}")
    for r in comparisons:
        if r['metric'] in ('net_pnl','max_drawdown'):
            lines.append(f"Paired difference vs {r['reference']} / {r['metric']}: {r['mean_diff']:+.4f} approx 95% [{r['approx_95_lower']:+.4f},{r['approx_95_upper']:+.4f}]")
    lines.append('Selection and screens are finite-sample and conditional on this generator, including the assumed hedge venue.')
    report='\n'.join(lines)+'\n'
    (out_dir/'results.txt').write_text(report, encoding='utf8', newline='\n')
    print(report)
    return {'selected':selected,'ranking':ranking,'validation':checks,'summary':table,'regimes':regime_table,'comparisons':comparisons,'sensitivity':sens}
if __name__=='__main__':
    run()
