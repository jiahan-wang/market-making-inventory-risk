"""Stage 5 figures. Needs matplotlib; run after risk_budget_optimizer.py."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent/'output'

def load(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def main():
    train=load('training_candidate_grid.csv')
    summary=load('test_strategy_summary.csv')
    regimes=load('test_by_market_regime.csv')
    fig,ax=plt.subplots(figsize=(9.2,5.5))
    for group,marker in [('A_fixed','o'),('B_inventory','s'),('C_vol_adaptive','^')]:
        vals=[r for r in train if r['strategy']==group]
        ax.scatter([float(r['mean_max_drawdown']) for r in vals],
                   [float(r['mean_net_pnl']) for r in vals],
                   marker=marker,s=65,alpha=.78,label=group)
    ax.axvline(1,ls='--',color='gray',label='Mean DD budget = 1.0')
    chosen=next(r for r in train if r['strategy']=='C_vol_adaptive' and r['base_half_spread']=='0.09' and r['inventory_alpha']=='0.016' and r['volatility_beta']=='0.4')
    ax.scatter([float(chosen['mean_max_drawdown'])],[float(chosen['mean_net_pnl'])],
               s=300,facecolors='none',edgecolors='black',linewidths=1.5,label='Budget-selected strategy')
    ax.set(xlabel='Average maximum drawdown (training scenarios)',ylabel='Average net P&L (training scenarios)',title='Strategy candidate grid: return versus drawdown')
    ax.grid(alpha=.2);ax.legend(fontsize=9);fig.tight_layout()
    fig.savefig(ROOT/'training_risk_return_scatter.png',dpi=175)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9.2,5.1))
    regime_labels=['calm','normal','stressed']
    labels=[('budget_selected','Risk-budget'),('train_pnl_max','Highest training P&L'),('fixed_baseline','Fixed spread'),('stage4_original','Prior conservative')]
    for i,(name,short) in enumerate(labels):
        vals=[next(float(r['mean_max_drawdown']) for r in regimes if r['label']==name and r['regime']==g) for g in regime_labels]
        xs=[j-0.30+i*0.20 for j in range(3)]
        ax.bar(xs, vals, width=.19,label=short)
    ax.set_xticks(range(3),['Calm','Normal','Stressed'])
    ax.axhline(1, ls='--',c='gray',label='Overall mean DD budget (not regime-specific)')
    ax.set(ylabel='Average maximum drawdown',title='Test-set drawdown by market regime')
    ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(ROOT/'test_regime_drawdowns.png',dpi=175)
    plt.close(fig)
    print('Saved figures to',ROOT)

if __name__=='__main__': main()
