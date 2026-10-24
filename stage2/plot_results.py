"""Stage 2 figures. Needs matplotlib; reads the CSVs in output/."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'

def read_csv(name):
    with (OUT / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

rows = read_csv('rounds_two_strategies.csv')
records = read_csv('all_200_scenarios.csv')
for key, ylabel, filename in [
    ('equity_after', 'Marked equity (synthetic units)', 'equity_paths.png'),
    ('inventory_after', 'Inventory (units)', 'inventory_paths.png'),
]:
    plt.figure(figsize=(9, 4.6))
    for strategy, name in [('fixed','A: Fixed quote'),('inventory_aware','B: Inventory-aware')]:
        selected = [r for r in rows if r['strategy'] == strategy]
        xs=[0]+[int(r['round']) for r in selected]
        ys=[0]+[float(r[key]) for r in selected]
        plt.plot(xs,ys,label=name,linewidth=1.8)
    plt.xlabel('Round')
    plt.ylabel(ylabel)
    plt.title('One matched synthetic market (seed 2026)')
    plt.grid(alpha=0.20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=180)
    plt.close()

plt.figure(figsize=(9, 4.6))
for strategy, name in [('fixed','A: Fixed quote'),('inventory_aware','B: Inventory-aware')]:
    selected=[float(r['ending_pnl_after_liquidation']) for r in records if r['strategy']==strategy]
    plt.hist(selected,bins=25,alpha=0.55,label=name)
plt.xlabel('Final P&L after assumed liquidation (synthetic units)')
plt.ylabel('Number of scenarios')
plt.title('200 matched synthetic market scenarios (100 rounds each)')
plt.legend()
plt.tight_layout()
plt.savefig(OUT/'pnl_distributions.png',dpi=180)
plt.close()
print('Saved three charts to', OUT)
