import platform, resource
import numpy as np
import scipy
import scipy.sparse as sp
import networkx as nx
from multiprocessing import get_context
from scipy.io import mmread
from scipy.sparse.linalg import spsolve
from time import perf_counter

FACEBOOK = ['Caltech36', 'Reed98', 'Haverford76', 'Simmons81',
            'Swarthmore42', 'Amherst41', 'Bowdoin47', 'Hamilton46']
STAGES = ['eig', 'form M', 'solve', 'reconstruct']

def adjacency(G):
    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.to_numpy_array(G.subgraph(max(nx.connected_components(G), key=len)))

def erdos_renyi(N):
    return adjacency(nx.gnp_random_graph(N, 40 / N, seed=0))

def facebook(name):
    return adjacency(nx.from_scipy_sparse_array(mmread(f'data/networks/socfb-{name}.mtx')))

def email():
    return adjacency(nx.read_edgelist('data/email/email-Eu-core.txt', nodetype=int))

def transition_matrix(W):
    return W / W.sum(1)[:, None]

def meeting_times(W):
    '''Algorithm 1 for F = 11^T - I and h = 0 (pair_poisson of algorithm.py), timed by stage.'''
    N = len(W)
    clock = [perf_counter()]

    # stage 1: eigendecomposition of Ptilde = Pi^{1/2} P Pi^{-1/2}
    pi = W.sum(1) / W.sum()
    sqpi = np.sqrt(pi)
    P = transition_matrix(W)
    Ptilde = sqpi[:, None] * P / sqpi[None, :]
    lam, V = np.linalg.eigh(0.5 * (Ptilde + Ptilde.T))
    idx = np.argsort(-lam)
    lam, V = lam[idx], V[:, idx]
    clock.append(perf_counter())

    # stage 2: divisors S of Eq. (7) and M of Eq. (11), entry-by-entry
    S = 1.0 - 0.5 * (lam[:, None] + lam[None, :])
    S[0, 0] = 1.0
    S = 1.0 / S
    M = np.empty((N, N))
    for i in range(N):
        U = V[i] * V
        M[i] = np.sum(U * (U @ S), axis=1)
    clock.append(perf_counter())

    # stage 3: Fhat_V and f, then solve Eq. (12) for Hhat_V_11 and d
    F = np.ones((N, N)) - np.eye(N)
    Fhat_V = V.T @ (np.outer(sqpi, sqpi) * F) @ V
    f = np.einsum('ij,jk,ik->i', V, Fhat_V * S, V)
    A = np.zeros((N + 1, N + 1))
    A[0, 1:] = pi ** 2
    A[1:, 0] = pi
    A[1:, 1:] = M * pi[None, :]
    sol = np.linalg.solve(A, np.concatenate([[-Fhat_V[0, 0]], -f]))
    Hhat_V_11, d = sol[0], sol[1:]
    clock.append(perf_counter())

    # stage 4: Dhat_V and Hhat_V, then reconstruct H
    Dhat_V = np.einsum('i,ij,ik->jk', pi * d, V, V)
    Hhat_V = (Fhat_V + Dhat_V) * S
    Hhat_V[0, 0] = Hhat_V_11
    H = (V @ Hhat_V @ V.T) / np.outer(sqpi, sqpi)
    clock.append(perf_counter())
    return H, np.diff(clock)

def residual(tau, W):
    '''relative residual of Eq. (1).'''
    P = transition_matrix(W)
    R = 1 + 0.5 * (P @ tau + tau @ P.T) - tau
    np.fill_diagonal(R, 0)
    return np.abs(R).max() / tau.max()

def meeting_times_sparse(W):
    '''Eq. (1) solved directly as a sparse linear system over the ordered pairs (SuperLU).'''
    N = len(W)
    P, I = sp.csr_matrix(transition_matrix(W)), sp.identity(N, format='csr')
    A = sp.identity(N * N, format='csr') - 0.5 * (sp.kron(I, P) + sp.kron(P, I))
    off = ~np.eye(N, dtype=bool).ravel()
    tau = np.zeros(N * N)
    tau[off] = spsolve(A[off][:, off].tocsc(), np.ones(off.sum()))
    return tau.reshape(N, N)

def peak_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss // 2**20 if platform.system() == 'Darwin' else rss // 2**10

def run(W):
    tau, times = meeting_times(W)
    return (*times, times.sum(), peak_mb(), residual(tau, W))

def run_sparse(W):
    tau, times = meeting_times(W)
    t0 = perf_counter()
    tau_sparse = meeting_times_sparse(W)
    return (times.sum(), perf_counter() - t0, peak_mb(), np.abs(tau - tau_sparse).max() / tau.max())

def show(label, *cells):
    cells = [f'{c:.4g}' if isinstance(c, float) else c for c in cells]
    print(f'{label:14}' + ''.join(f'{c:>12}' for c in cells), flush=True)

if __name__ == '__main__':
    print(platform.platform(), platform.processor(), 'Python', platform.python_version(),
          'NumPy', np.__version__, 'SciPy', scipy.__version__, flush=True)
    graphs = ([('Erdos-Renyi', erdos_renyi(N)) for N in [250, 500, 1000, 2000, 4000]]
            + [(name, facebook(name)) for name in FACEBOOK]
            + [('Email-Eu-core', email())])

    # a fresh process per graph, so that peak memory is per graph
    with get_context('spawn').Pool(1, maxtasksperchild=1) as pool:
        show('graph', 'N', '|E|', *STAGES, 'total', 'memory (MB)', 'residual')
        for label, W in graphs:
            show(label, len(W), int(W.sum() / 2), *pool.apply(run, (W,)))
        print()
        show('graph', 'N', '|E|', 'Algorithm 1', 'sparse', 'memory (MB)', 'difference')
        for N in [100, 150]:
            W = erdos_renyi(N)
            show('Erdos-Renyi', len(W), int(W.sum() / 2), *pool.apply(run_sparse, (W,)))
