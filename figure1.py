import os, sys, glob
import numpy as np
import scipy.io as sio
import networkx as nx
import matplotlib.pyplot as plt
from time import perf_counter
from algorithm import bc_stars

KINDS = ['pp', 'ff', 'pf']
COLORS = {'pp': '#990099', 'ff': '#009999', 'pf': '#BF9000'}
NPZ = 'data/fb100_results.npz'

def load_networks(directory='data/networks'):
    '''load all .mtx files, return list of (name, Graph) sorted by N.'''
    graphs = []
    for f in sorted(glob.glob(os.path.join(directory, '*.mtx'))):
        name = os.path.basename(f).replace('socfb-', '').replace('.mtx', '')
        G = nx.from_scipy_sparse_array(sio.mmread(f))
        G.remove_edges_from(nx.selfloop_edges(G))
        cc = max(nx.connected_components(G), key=len)
        G = nx.convert_node_labels_to_integers(G.subgraph(cc).copy())
        graphs.append((name, G))
    return sorted(graphs, key=lambda x: x[1].number_of_nodes())

def compute(graphs):
    '''compute (b/c)* for each graph, caching incrementally.'''
    cached = {}
    if os.path.exists(NPZ):
        cached = dict(np.load(NPZ, allow_pickle=True)['results'].item())

    results = dict(cached)
    for name, G in graphs:
        if name in results:
            print(f'  {name}: cached', flush=True)
            continue
        N, E = G.number_of_nodes(), G.number_of_edges()
        print(f'  {name} (N={N}, avg deg={2*E/N:.1f})...', end=' ', flush=True)
        t0 = perf_counter()
        vals = bc_stars(nx.to_numpy_array(G, dtype=float))
        print(f'done [{perf_counter()-t0:.1f}s]', flush=True)
        results[name] = {'N': N, 'E': E, 'avg_deg': 2*E/N, **vals}
        np.savez(NPZ, results=results)

    return results

def plot(results):
    names = sorted(results.keys(), key=lambda n: results[n]['N'])
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for j, n in enumerate(names):
        vals = [results[n][k] for k in KINDS]
        ax.plot([min(vals), max(vals)], [j, j], color='#cccccc', lw=1, zorder=1)

    for k in KINDS:
        ax.scatter([results[n][k] for n in names], y, s=45, color=COLORS[k],
                   edgecolors='white', linewidths=0.4, zorder=3)

    for k in KINDS:
        ax.text(results[names[0]][k], -0.45, k, ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=COLORS[k])

    ax.set_xscale('log')
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax2 = ax.secondary_yaxis('right')
    ax2.set_yticks(y)
    ax2.set_yticklabels(
        [f'$N={results[n]["N"]},\\,\\bar{{k}}={results[n]["avg_deg"]:.0f}$'
         for n in names], fontsize=7)
    ax2.tick_params(length=3, width=0.6, direction='out')
    ax2.spines['right'].set_linewidth(0.6)
    ax.set_xlabel(r'$(b/c)^*$', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig('Fig1.pdf', bbox_inches='tight', pad_inches=0.05)
    print('Saved Fig1.pdf')

if __name__ == '__main__':
    if '--plot-only' in sys.argv:
        results = dict(np.load(NPZ, allow_pickle=True)['results'].item())
    else:
        results = compute(load_networks())
    plot(results)
