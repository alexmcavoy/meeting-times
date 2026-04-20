import numpy as np

def pair_poisson(P, F, h, alpha=0.5):
    '''solve the Poisson equation for the absorbing lazy pair walk.'''
    N = P.shape[0]
    pi = np.linalg.solve(np.eye(N) - P.T + np.ones((N, N)), np.ones(N))
    sqpi = np.sqrt(pi)

    # eigendecompose Ptilde = Pi^{1/2} P Pi^{-1/2}
    Ptilde = sqpi[:, None] * P / sqpi[None, :]
    lam, V = np.linalg.eigh(0.5 * (Ptilde + Ptilde.T))
    idx = np.argsort(-lam)
    lam, V = lam[idx], V[:, idx]

    # form divisor matrix S
    S = 1.0 - alpha * (lam[:, None] + lam[None, :])
    degenerate = abs(S[0, 0]) < 1e-12
    if degenerate:
        S[0, 0] = 1.0
    S = 1.0 / S

    # Fhat_V = V^T Pi^{1/2} F Pi^{1/2} V
    Fhat_V = V.T @ (np.outer(sqpi, sqpi) * F) @ V

    # form correction vectors f and M
    f = np.einsum('ij,jk,ik->i', V, Fhat_V * S, V)
    M = np.empty((N, N))
    for i in range(N):
        U = V[i] * V
        M[i] = np.sum(U * (U @ S), axis=1)

    # solve correction system
    if degenerate:
        A = np.zeros((N + 1, N + 1))
        A[0, 1:] = pi ** 2
        A[1:, 0] = pi
        A[1:, 1:] = M * pi[None, :]
        rhs = np.empty(N + 1)
        rhs[0] = -Fhat_V[0, 0]
        rhs[1:] = pi * h - f
        sol = np.linalg.solve(A, rhs)
        Hhat_V_11, d = sol[0], sol[1:]
    else:
        d = np.linalg.solve(M * pi[None, :], pi * h - f)

    # back-transform
    Dhat_V = np.einsum('i,ij,ik->jk', pi * d, V, V)
    Hhat_V = (Fhat_V + Dhat_V) * S
    if degenerate:
        Hhat_V[0, 0] = Hhat_V_11

    return (V @ Hhat_V @ V.T) / np.outer(sqpi, sqpi)

def bc_stars(W, kinds=('pp', 'ff', 'pf')):
    '''critical (b/c)* ratios for death-Birth updating with accumulated payoffs.'''
    N = W.shape[0]
    deg = W.sum(1)
    pi = deg / deg.sum()
    P = W / deg[:, None]
    P2 = P @ P
    tau = pair_poisson(P, np.ones((N, N)) - np.eye(N), np.zeros(N))
    coefs = {'pp': (W, W), 'ff': (P, P), 'pf': (W, P)}
    out = {}
    for k in kinds:
        beta, gamma = coefs[k]
        s_g = np.einsum('i,ij,ij,jl->', pi, P2, tau, gamma)
        s_b = (np.einsum('i,ij,il,lj->', pi, P2, tau, beta)
             - np.einsum('i,il,li->',    pi, tau, beta))
        out[k] = s_g / s_b
    return out
