import os, sys, gzip, urllib.request
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from time import perf_counter
from algorithm import bc_stars

EDGES_URL = 'https://snap.stanford.edu/data/email-Eu-core.txt.gz'
EDGES_FILE = 'data/email/email-Eu-core.txt'
NPZ_DATA = 'data/email/email_data.npz'
NPZ_LAYOUT = 'data/email/email_layout.npz'
N_NULLS = 100
KINDS = ['pp', 'ff', 'pf']
COLORS = {'pp': '#990099', 'ff': '#009999', 'pf': '#BF9000'}

def largest_cc(G):
    '''return largest connected component with relabeled nodes.'''
    cc = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(cc).copy())

def load_graph():
    '''download (if needed) and return largest connected component.'''
    if not os.path.exists(EDGES_FILE):
        print('Downloading edge list...', flush=True)
        with urllib.request.urlopen(EDGES_URL) as r, open(EDGES_FILE, 'wb') as f:
            f.write(gzip.decompress(r.read()))
    G = nx.read_edgelist(EDGES_FILE, nodetype=int, create_using=nx.Graph())
    G.remove_edges_from(nx.selfloop_edges(G))
    return largest_cc(G)

def load_cache():
    '''Load cached (b/c)* results, or return (None, None).'''
    if not os.path.exists(NPZ_DATA):
        return None, None
    d = np.load(NPZ_DATA)
    try:
        return ({k: float(d[f'real_{k}']) for k in KINDS},
                {k: list(d[f'null_{k}']) for k in KINDS})
    except KeyError:
        return None, None

def save_cache(real, nulls):
    np.savez(NPZ_DATA, **{f'real_{k}': real[k] for k in KINDS},
             **{f'null_{k}': np.array(nulls[k]) for k in KINDS})

def compute(G):
    '''Compute (b/c)* for real network + N_NULLS rewirings, with caching.'''
    M = G.number_of_edges()
    real, nulls = load_cache()

    if real is None:
        print('Computing (b/c)* on real network...', flush=True)
        real = bc_stars(nx.to_numpy_array(G, dtype=float))
        nulls = {k: [] for k in KINDS}
        save_cache(real, nulls)

    for i in range(len(nulls['pp']), N_NULLS):
        H = G.copy()
        nx.double_edge_swap(H, nswap=10*M, max_tries=1000*M, seed=i+1)
        W = nx.to_numpy_array(largest_cc(H), dtype=float)
        t0 = perf_counter()
        vals = bc_stars(W)
        for k in KINDS:
            nulls[k].append(vals[k])
        save_cache(real, nulls)
        print(f'  null {i+1}/{N_NULLS}  [{perf_counter()-t0:.1f}s]', flush=True)

    return real, nulls

def load_layout(G):
    '''Kamada-Kawai + spring layout (arrays), cached to disk.'''
    if os.path.exists(NPZ_LAYOUT):
        d = np.load(NPZ_LAYOUT)
        if set(int(u) for u in d['nodes']) == set(G.nodes()):
            return d['xs'].copy(), d['ys'].copy()

    print('Computing layout...', flush=True)
    pos = nx.spring_layout(G, pos=nx.kamada_kawai_layout(G), iterations=50,
                           k=1.5/np.sqrt(G.number_of_nodes()), seed=0)
    nodes = sorted(pos)
    xs = np.array([pos[u][0] for u in nodes])
    ys = np.array([pos[u][1] for u in nodes])
    xs -= xs.mean(); ys -= ys.mean()
    s = max(xs.std(), ys.std()); xs /= s; ys /= s
    np.savez(NPZ_LAYOUT, nodes=nodes, xs=xs, ys=ys)
    return xs, ys

def draw_network(ax, G, xs, ys):
    cmap = mcolors.LinearSegmentedColormap.from_list(
        '_', ['#7fa9c8', '#74b3ad', '#88b577', '#d4b85a', '#cf805c', '#b56a8a'])
    degs = np.array([G.degree(u) for u in sorted(G.nodes())], dtype=float)
    dlog = np.log1p(degs)
    dlog = (dlog - dlog.min()) / (dlog.max() - dlog.min())

    xy = np.column_stack([xs, ys])
    segs = np.array([[xy[u], xy[v]] for u, v in G.edges()])
    ax.add_collection(LineCollection(segs, colors=[(0.15, 0.12, 0.20, 0.25)],
                                     linewidths=0.22, zorder=1))

    ax.scatter(xs, ys, s=3+30*dlog, c=cmap(dlog),
               edgecolors='black', linewidths=0.30, zorder=3)
    ax.set_aspect('equal'); ax.margins(0.05)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

def draw_strip(ax, k, real_val, null_vals, color):
    ax.scatter(null_vals, np.zeros_like(null_vals), s=14, color=color,
               alpha=0.45, edgecolors='none', zorder=3)
    ax.scatter([real_val], [0], s=110, color=color, marker='D',
               edgecolors='white', linewidths=1.0, zorder=4)
    lo, hi = min(null_vals.min(), real_val), max(null_vals.max(), real_val)
    pad = 0.10 * (hi - lo) if hi > lo else max(0.05 * abs(hi), 1e-9)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(-0.5, 0.5); ax.set_yticks([])
    for sp in ('left', 'right', 'top'):
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['bottom'].set_linewidth(0.6)
    ax.tick_params(axis='x', direction='out', length=3, width=0.6)
    ax.text(1.02, 0.5, k, transform=ax.transAxes, ha='left', va='center',
            fontsize=11, fontweight='bold', color=color)

if __name__ == '__main__':
    G = load_graph()
    if '--plot-only' in sys.argv:
        real, nulls = load_cache()
        assert real is not None, 'No cached data; run without --plot-only first.'
    else:
        real, nulls = compute(G)

    xs, ys = load_layout(G)
    fig = plt.figure(figsize=(9.5, 3.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 3], hspace=1.0, wspace=0.00,
                          left=0.005, right=0.93, top=0.94, bottom=0.08)
    draw_network(fig.add_subplot(gs[:, 0]), G, xs, ys)
    for row, k in enumerate(KINDS):
        draw_strip(fig.add_subplot(gs[row, 1]), k, real[k],
                   np.asarray(nulls[k]), COLORS[k])
    fig.savefig('Fig2.pdf', bbox_inches='tight', pad_inches=0.02)
    print('Saved Fig2.pdf')
