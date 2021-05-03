"""
Utilities and helper functions.

These functions can be used to construct 2-sided 1 - alpha confidence
bounds for the average treatment effect in a randomized experiment with binary
outcomes and two treatments.
"""

from itertools import combinations
from math import comb, floor
import numpy as np


def nchoosem(n, m):
    """
    Exact re-randomization matrix for small n choose m.

    Parameters
    ----------
    n: int
        total number of subjects
    m: int
        number of subjects under treatment

    Returns
    -------
    Z: list
        re-randomization matrix
    """
    c = comb(n, m)
    trt = combinations(np.arange(1, n+1), m)
    Z = [[None]*n for i in np.arange(c)]
    for i in np.arange(c):
        co = next(trt)
        Z[i] = [1 if j in co else 0 for j in np.arange(1, n+1)]
    return Z


def combs(n, m, nperm):
    """
    Sample from re-randomization matrix.

    Parameters
    ----------
    n: int
        total number of subjects
    m: int
        number of subjects under treatment
    nperm: int
        number of permutations

    Returns
    -------
    Z: list
        sample from re-randomization matrix
    """
    trt = [[None]*m for i in np.arange(nperm)]
    Z = [[None]*n for i in np.arange(nperm)]
    for i in np.arange(nperm):
        trt[i] = np.random.choice(n, m, replace=False).tolist()
    for i in np.arange(nperm):
        Z[i] = [1 if j in trt[i] else 0 for j in np.arange(n)]
    return Z


def pval_one_lower(n, m, N, Z_all, tau_obs):
    """
    Calculate p-value for method I.

    Parameters
    ----------
    n: int
        total number of subjects
    m: int
        number of subjects under treatment
    N: list
        potential table
    Z_all: list
        re-randomization or sample of re-randomization matrix
    tau_obs: float
        observed value of tau

    Returns
    -------
    pl: float
        p-value
    """
    n_Z_all = len(Z_all)
    dat = [[None]*2 for i in np.arange(n)]
    if N[0] > 0:
        for i in np.arange(N[0]):
            dat[i] = [1]*2
    if N[1] > 0:
        for i in np.arange(N[0], N[0]+N[1]):
            dat[i][0] = 1
            dat[i][1] = 0
    if N[2] > 0:
        for i in np.arange(N[0]+N[1], N[0]+N[1]+N[2]):
            print(i)
            dat[i][0] = 0
            dat[i][1] = 1
    if N[3] > 0:
        for i in np.arange(N[0]+N[1]+N[2], sum(N)):
            dat[i] = [0]*2
    x = [i[0]/m for i in dat]
    y = [i[1]/(n-m) for i in dat]
    a = []
    b = []
    for i in np.arange(len(Z_all)):
        a.append(sum([x[j]*Z_all[i][j] for j in np.arange(len(x))]))
        b.append(sum([(1-Z_all[i][j])*y[j] for j in np.arange(len(y))]))
    tau_hat = [a[i]-b[i] for i in np.arange(len(a))]
    if isinstance(tau_obs, list):
        count = sum([1 if round(tau_hat[i], 15) >= round(tau_obs[i], 15) else 0
                     for i in np.arange(len(tau_hat))])
    else:
        count = sum([1 if round(tau_hat[i], 15) >= round(tau_obs, 15) else 0
                     for i in np.arange(len(tau_hat))])
    pl = count/n_Z_all
    return pl


def pval_two(n, m, N, Z_all, tau_obs):
    """
    Calculate p-value for method 3.

    Parameters
    ----------
    n: int
        total number of subjects
    m: int
        number of subjects under treatment
    N: list
        potential table
    Z_all: list
        re-randomization or sample of re-randomization matrix
    tau_obs: float
        observed value of tau

    Returns
    -------
    pl: float
        p-value
    """
    n_Z_all = len(Z_all)
    dat = [[None]*2 for i in np.arange(n)]
    if N[0] > 0:
        for i in np.arange(N[0]):
            dat[i] = [1]*2
    if N[1] > 0:
        for i in np.arange(N[0], N[0]+N[1]):
            dat[i][0] = 1
            dat[i][1] = 0
    if N[2] > 0:
        for i in np.arange(N[0]+N[1], N[0]+N[1]+N[2]):
            dat[i][0] = 0
            dat[i][1] = 1
    if N[3] > 0:
        for i in np.arange(N[0]+N[1]+N[2], sum(N)):
            dat[int(i)] = [0]*2
    x = [i[0]/m for i in dat]
    y = [i[1]/(n-m) for i in dat]
    a = []
    b = []
    for i in np.arange(len(Z_all)):
        a.append(sum([x[j]*Z_all[i][j] for j in np.arange(len(x))]))
        b.append(sum([(1-Z_all[i][j])*y[j] for j in np.arange(len(y))]))
    tau_hat = [a[i]-b[i] for i in np.arange(len(a))]
    tau_N = (N[1]-N[2])/n
    count = sum([1 if round(abs(tau_hat[i]-tau_N), 14) >=
                 round(abs(tau_obs-tau_N), 14) else 0
                 for i in np.arange(len(tau_hat))])
    pd = count/n_Z_all
    return pd


def check_compatible(n11, n10, n01, n00, N11, N10, N01):
    """
    Check that observed table is compatible with potential table.

    Parameters
    ----------
    n11: int
        number of subjects under treatment that experienced outcome 1
    n10: int
        number of subjects under treatment that experienced outcome 0
    n01: int
        number of subjects under control that experienced outcome 1
    n00: int
        number of subjects under control that experienced outcome 0
    N11: list of integers
        potential number of subjects under treatment that experienced
        outcome 1
    N10: list of integers
        potential number of subjects under treatment that experienced
        outcome 0
    N01: list of integers
        potential number of subjects under treatment that experienced
        outcome 0

    Returns
    -------
    compact: list
        booleans indicating compatibility of inputs
    """
    n = n11+n10+n01+n00
    n_t = len(N10)
    lefts = [[] for i in np.arange(len(N10))]
    rights = [[] for i in np.arange(len(N10))]
    for i in np.arange(len(N10)):
        lefts[i].append(0)
        rights[i].append(N11[i])
        lefts[i].append(n11 - N10[i])
        rights[i].append(n11)
        lefts[i].append(N11[i]-n01)
        rights[i].append(N11[i]+N01[i]-n01)
        lefts[i].append(N11[i]+N01[i]-n10-n01)
        rights[i].append(n-N10[i]-n01-n10)
    left = [max(x) for x in lefts]
    right = [min(x) for x in rights]
    compact = [left[i] <= right[i] for i in np.arange(len(left))]
    return compact


def tau_lower_N11_oneside(n11, n10, n01, n00, N11, Z_all, alpha):
    """
    Calculate tau_min and N_accept for method I.

    Parameters
    ----------
    n11: int
        number of subjects under treatment that experienced outcome 1
    n10: int
        number of subjects under treatment that experienced outcome 0
    n01: int
        number of subjects under control that experienced outcome 1
    n00: int
        number of subjects under control that experienced outcome 0
    N11: int
        potential number of subjects under treatment that experienced
        outcome 1
    Z_all: list
        re-randomization or sample of re-randomization matrix
    alpha: float
        1 - confidence level

    Returns
    -------
    tau_min: float
        minimum tau value of accepted potential tables
    N_accept:
        accepted potential table
    """
    n = n11+n10+n01+n00
    m = n11+n10
    N01 = 0
    N10 = 0
    tau_obs = n11/m - n01/(n-m)
    M = [0]*(n-N11+1)
    while (N10 <= (n - N11 - N01)) & (N01 <= (n - N11)):
        pl = pval_one_lower(n, m, [N11, N10, N01, n - (N11 + N10 + N01)],
                            Z_all, tau_obs)
        if pl >= alpha:
            M[N01] = N10
            N01 += 1
        else:
            N10 += 1
    if N01 <= (n - N11):
        for i in np.arange(N01, (n-N11+1)):
            M[i] = n+1
    N11_vec0 = [N11]*(n-N11+1)
    N10_vec0 = M
    N01_vec0 = np.arange((n-N11+1)).tolist()
    N11_vec = []
    N10_vec = []
    N01_vec = []
    for i in np.arange(len(N11_vec0)):
        if N10_vec0[i] <= (n - N11_vec0[i] - N01_vec0[i]):
            N10_vec.extend(np.arange(N10_vec0[i], n-N11_vec0[i]-N01_vec0[i]+1)
                           .tolist())
            N11_vec.extend([N11_vec0[i]]*(n-N11_vec0[i]-N01_vec0[i]-N10_vec0[i]
                                          + 1))
            N01_vec.extend([N01_vec0[i]]*(n-N11_vec0[i]-N01_vec0[i]-N10_vec0[i]
                                          + 1))
    compat = check_compatible(n11, n10, n01, n00, N11_vec, N10_vec, N01_vec)
    if sum(compat) > 0:
        N10_vec_compat = [N10_vec[i] for i in np.arange(len(N10_vec)) if
                          compat[i]]
        N01_vec_compat = [N01_vec[i] for i in np.arange(len(N10_vec)) if
                          compat[i]]
        tau_min = min([N10_vec_compat[i] - N01_vec_compat[i] for i in
                       np.arange(len(N10_vec_compat))])/n
        accept_pos = [i for i in np.arange(len(N10_vec_compat)) if
                      N10_vec_compat[i]-N01_vec_compat[i] == n*tau_min]
        accept_pos = accept_pos[0]
        N_accept = [N11, N10_vec_compat[accept_pos],
                    N01_vec_compat[accept_pos],
                    n-(N11+N10_vec_compat[accept_pos] +
                       N01_vec_compat[accept_pos])]
    else:
        tau_min = (n11 + n00)/n
        N_accept = float('NaN')
    return tau_min, N_accept
