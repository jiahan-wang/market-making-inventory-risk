"""Stage 3 figures. Needs matplotlib; reads the CSVs in output/."""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

DIR = Path(__file__).resolve().parent / 'output'
LABELS = {'A_fixed': 'A Fixed quotes', 'B_inventory': 'B Inventory-aware', 'C_vol_adaptive': 'C Volatility-adaptive'}


def rows(name):
    with (DIR/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save(name):
    plt.tight_layout()
    plt.savefig(DIR/name, dpi=180, bbox_inches='tight')
    plt.close()


def risk_return():
    data = rows('test_summary.csv')
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    markers = {'calm': 'o', 'normal': 's', 'stressed': '^'}
    for regime in ('calm', 'normal', 'stressed'):
        for r in data:
            if r['regime'] != regime: continue
            x, y = float(r['avg_max_drawdown']), float(r['avg_net_pnl'])
            ax.scatter(x, y, s=90, marker=markers[regime])
            ax.annotate(f"{r['strategy'][0]} ({regime})", (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set(xlabel='Mean maximum drawdown (lower is better)', ylabel='Mean net P&L (higher is better)',
           title='Trading strategies: return vs. risk across synthetic market regimes')
    ax.grid(alpha=0.25)
    save('risk_return_tradeoff.png')


def demo_paths():
    raw = rows('one_stressed_market_rounds.csv')
    groups = defaultdict(list)
    for r in raw:
        groups[r['strategy']].append(r)
    for field, filename, label in [
        ('equity_after', 'stressed_example_equity.png', 'Marked-to-mid account equity'),
        ('inventory_after', 'stressed_example_inventory.png', 'Net inventory (contracts)')
    ]:
        fig, ax = plt.subplots(figsize=(9, 4.7))
        for strategy, grp in groups.items():
            ax.plot([int(r['round']) for r in grp], [float(r[field]) for r in grp], label=LABELS[strategy], linewidth=1.65)
        ax.axhline(0, linewidth=0.8, alpha=0.5)
        ax.set(xlabel='Trading round', ylabel=label,
               title='One stressed market, seed 2026')
        ax.grid(alpha=0.20)
        ax.legend()
        save(filename)


def markouts():
    raw = [r for r in rows('test_summary.csv') if r['regime'] == 'stressed']
    names = [LABELS[r['strategy']] for r in raw]
    positions = list(range(len(raw)))
    w=0.34
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar([p-w/2 for p in positions], [float(r['avg_informed_markout']) for r in raw], width=w, label='Informed customer')
    ax.bar([p+w/2 for p in positions], [float(r['avg_noise_markout']) for r in raw], width=w, label='Uninformed customer')
    ax.set_xticks(positions, names)
    ax.set(ylabel='One-step fill markout (higher is better)',
           title='Post-trade adverse selection in stressed simulated markets')
    ax.axhline(0, linewidth=0.8)
    ax.legend()
    save('adverse_selection_markouts.png')


if __name__=='__main__':
    risk_return()
    demo_paths()
    markouts()
    print('Saved 4 graphs to output/')
