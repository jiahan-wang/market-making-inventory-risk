"""Stage 7 figures. Needs matplotlib; reads saved tables, runs no simulation."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / 'output'


def load(name):
    with (OUT / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


REGIME_COLOURS = {'calm': '#4c72b0', 'normal': '#dd8452', 'stressed': '#c44e52'}
LABELS = {'stage7_selected': 'Selected (hedge above 1)',
          'stage6_risk_only': 'Risk-only reference',
          'stage5_reference': 'Unhedged reference'}

# The whole distribution, not just the three means: a rule that holds on average
# can still contain markets that lose a lot, and the left panel is where that shows.
markets = [r for r in load('test_360_raw.csv') if r['label'] == 'stage7_selected']
summary = load('test_360_policy_summary.csv')

fig, (left, right) = plt.subplots(1, 2, figsize=(11.6, 4.9))

for regime, colour in REGIME_COLOURS.items():
    xs = [float(r['max_drawdown']) for r in markets if r['regime'] == regime]
    ys = [float(r['net_pnl']) for r in markets if r['regime'] == regime]
    left.scatter(xs, ys, s=13, alpha=0.6, color=colour, linewidths=0, label=regime)
left.axvline(1.0, ls='--', lw=1, color='0.35')
left.set_xlabel('maximum drawdown, one market')
left.set_ylabel('net P&L, one market')
left.set_title('Selected policy on 360 held-out markets')
left.legend(frameon=False, fontsize=8, loc='lower left')
left.grid(alpha=0.15)

PLACEMENT = {'stage7_selected': ((0, -17), 'center'),
             'stage5_reference': ((-9, 4), 'right'),
             'stage6_risk_only': ((9, -14), 'left')}

for row in summary:
    x = float(row['stressed_fill_rate']) * 100
    y = float(row['stressed_mean_max_drawdown'])
    right.scatter(x, y, s=70, zorder=3)
    offset, align = PLACEMENT[row['label']]
    right.annotate(LABELS[row['label']], (x, y), xytext=offset,
                   textcoords='offset points', fontsize=8, ha=align)
right.axvline(20, ls='--', lw=1, color='0.35')
right.axhline(1.0, ls='--', lw=1, color='0.35')
right.set_xlim(0, 42)
right.set_ylim(0.4, 1.75)
right.set_xlabel('stressed-market customer fill rate (%)')
right.set_ylabel('stressed mean max drawdown')
right.set_title('Screens: participation floor and drawdown budget')
right.grid(alpha=0.15)

fig.tight_layout()
fig.savefig(OUT / 'participation_vs_risk.png', dpi=170)
plt.close(fig)

rows = load('hedge_cost_sensitivity.csv')
x = [float(r['external_hedge_half_spread']) for r in rows]
pnl = [float(r['mean_net_pnl']) for r in rows]
dd = [float(r['stressed_mean_max_drawdown']) for r in rows]
low = [float(r['stressed_dd_ci_low']) for r in rows]
high = [float(r['stressed_dd_ci_high']) for r in rows]

fig, ax = plt.subplots(figsize=(8.8, 5))
ax.plot(x, pnl, marker='o', label='Mean net P&L')
ax.errorbar(x, dd, yerr=[[d - l for d, l in zip(dd, low)], [h - d for d, h in zip(dd, high)]],
            marker='s', capsize=4, lw=1, label='Stressed mean max drawdown (95% bootstrap)')
ax.axhline(1.0, ls='--', lw=1, color='0.35')
ax.set_xlabel('external hedge half-spread (simulated price units)')
ax.set_ylabel('simulated metric')
ax.set_title('Hedge cost sensitivity, fixed policy on the same 360 paths')
ax.legend(frameon=False, fontsize=8, loc='center left')
ax.grid(alpha=0.15)
fig.tight_layout()
fig.savefig(OUT / 'hedge_cost_sensitivity.png', dpi=170)
plt.close(fig)

grid_rows = load('dgp_sensitivity.csv')
mults = sorted({float(r['informed_share_multiplier']) for r in grid_rows})
accs = sorted({float(r['informed_accuracy']) for r in grid_rows})
grid = [[next(float(r['mean_net_pnl']) for r in grid_rows
              if float(r['informed_share_multiplier']) == m and float(r['informed_accuracy']) == a)
         for a in accs] for m in mults]
fails = set()
for m in mults:
    for a in accs:
        row = next(r for r in grid_rows
                   if float(r['informed_share_multiplier']) == m
                   and float(r['informed_accuracy']) == a)
        if row['screens_met_on_this_generator'] != 'True':
            fails.add((m, a))

fig, ax = plt.subplots(figsize=(7.6, 5.4))
img = ax.imshow(grid, cmap='coolwarm', origin='lower', aspect='auto',
                vmin=-max(abs(min(map(min, grid))), abs(max(map(max, grid)))),
                vmax=max(map(max, grid)))
ax.set_xticks(range(len(accs)), [f'{a:.2f}' for a in accs])
ax.set_yticks(range(len(mults)), [f'x{m:.1f}' for m in mults])
ax.set_xlabel('informed-order accuracy')
ax.set_ylabel('informed-order share, relative to the shipped mix')
ax.set_title('Selected policy under other generators')
ax.figure.colorbar(img, ax=ax, label='mean net P&L (simulated price units)')
for i, m in enumerate(mults):
    for j, a in enumerate(accs):
        mark = ' x' if (m, a) in fails else ''
        ax.text(j, i, f'{grid[i][j]:+.2f}{mark}', ha='center', va='center', fontsize=8)
ax.text(1.0, -0.18, 'x = screen not met', transform=ax.transAxes, ha='right', fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'dgp_sensitivity.png', dpi=170)
plt.close(fig)

print('Generated 3 figures.')
