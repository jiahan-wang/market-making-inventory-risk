"""Stage 6 figures. Reads the saved final-split tables and runs nothing."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent/'output'
summary=ROOT/'test_regime_summary.csv'
if not summary.is_file():
    print('No stage 6 selection on this run, so there are no figures to draw.')
    raise SystemExit(0)

with summary.open(encoding='utf-8-sig',newline='') as f:
    rows=list(csv.DictReader(f))

labels=sorted({r['label'] for r in rows})
regimes=['calm','normal','stressed']
metrics=[('mean_net_pnl','net_pnl','Mean net P&L'),('mean_max_drawdown','drawdown','Mean max drawdown'),
         ('mean_fill_rate','fill_rate','Mean fill rate')]
for key,short,title in metrics:
    fig,ax=plt.subplots(figsize=(8.2,4.6))
    width=0.8/(len(labels) or 1)
    for i,label in enumerate(labels):
        vals=[float(r[key]) for r in rows if r['label']==label]
        ax.bar([j+i*width for j in range(len(regimes))],vals,width=width,label=label)
    ax.set_xticks([j+width*(len(labels)-1)/2 for j in range(len(regimes))],regimes)
    ax.set_ylabel(title + ' (simulated price units)' if short != 'fill_rate' else 'Customer fill rate')
    ax.set_title(title + ' by market regime')
    ax.legend(fontsize=8)
    ax.grid(axis='y',alpha=.15)
    fig.tight_layout()
    fig.savefig(ROOT/f'regime_{short}_comparison.png',dpi=170)
    plt.close(fig)
print('Saved three charts.')
