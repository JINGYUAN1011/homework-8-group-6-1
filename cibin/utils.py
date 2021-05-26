"""
Utilities and helper functions.

These functions can be used to construct 2-sided 1 - alpha confidence
bounds for the average treatment effect in a randomized experiment with binary
outcomes and two treatments.
"""

import numpy as np
from scipy.stats import hypergeom
from itertools import combinations
from math import comb, floor


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
    Z: numpy array
        re-randomization matrix
    """
    c = comb(n, m)
    trt = combinations(np.arange(n), m)
    Z = np.zeros((c, n), dtype=int)
    for i in np.arange(c):
        co = next(trt)
        for j in np.arange(n):
            if j in co:
                Z[i, j] = 1
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
    Z: numpy array
        sample from re-randomization matrix
    """
    Z = np.zeros((nperm, n))
    for i in np.arange(nperm):
        trt = np.random.choice(n, m, replace=False)
        for j in np.arange(n):
            if j in trt:
                Z[i, j] = 1
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
    N: numpy array
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
    dat = np.zeros((n, 2), dtype=int)
    if N[0] > 0:
        dat[0:N[0], :] = 1
    if N[1] > 0:
        for i in np.arange(N[0], N[0]+N[1]):
            dat[i, 0] = 1
            dat[i, 1] = 0
    if N[2] > 0:
        for i in np.arange(N[0]+N[1], N[0]+N[1]+N[2]):
            dat[i, 0] = 0
            dat[i, 1] = 1
    if N[3] > 0:
        for i in np.arange(N[0]+N[1]+N[2], sum(N)):
            dat[i] = [0]*2
    tau_hat = np.matmul(Z_all, dat[:, 0]/m) - np.matmul((1 - Z_all),
                                                        dat[:, 1]/(n-m))
    pl = sum(np.round(tau_hat, 15) >= round(tau_obs, 15))/n_Z_all
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
    N: numpy array
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
    dat = np.zeros((n, 2), dtype=int)
    if N[0] > 0:
        dat[0:N[0], :] = 1
    if N[1] > 0:
        for i in np.arange(N[0], N[0]+N[1]):
            dat[i, 0] = 1
            dat[i, 1] = 0
    if N[2] > 0:
        for i in np.arange(N[0]+N[1], N[0]+N[1]+N[2]):
            dat[i, 0] = 0
            dat[i, 1] = 1
    if N[3] > 0:
        for i in np.arange(N[0]+N[1]+N[2], sum(N)):
            dat[i] = [0]*2
    tau_hat = np.matmul(Z_all, dat[:, 0]/m) - np.matmul((1 - Z_all),
                                                        dat[:, 1]/(n-m))
    tau_N = (N[1]-N[2])/n
    pd = sum(np.round(abs(tau_hat-tau_N), 15) >= round(abs(tau_obs-tau_N),
                                                       15))/n_Z_all
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
    N11: numpy array of integers
        number of subjects under control and treatment with potential outcome 1
        outcome 1
    N10: numpy array of integers
        potential number of subjects under treatment that experienced
        outcome 0
    N01: numpy array of integers
        potential number of subjects under treatment that experienced
        outcome 0

    Returns
    -------
    compact: list
        booleans indicating compatibility of inputs
    """
    n = n11+n10+n01+n00
    n_t = len(N10)
    lefts = np.empty((n_t, 4), dtype=int)
    lefts[:, 0] = 0
    lefts[:, 1] = n11-N10
    lefts[:, 2] = N11-n01
    lefts[:, 3] = N11+N01-n10-n01
    rights = np.empty((n_t, 4), dtype=int)
    rights[:, 0] = N11
    rights[:, 1] = n11
    rights[:, 2] = N11+N01-n01
    rights[:, 3] = n-N10-n01-n10
    left = np.max(lefts, axis=1)
    right = np.min(rights, axis=1)
    compact = left <= right
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
        number of subjects under control and treatment with potential outcome 1
    Z_all: numpy array
        re-randomization or sample of re-randomization matrix
    alpha: float
        1 - confidence level

    Returns
    -------
    tau_min: float
        minimum tau value of accepted potential tables
    N_accept: numpy array
        accepted potential table
    """
    n = n11+n10+n01+n00
    m = n11+n10
    N01 = 0
    N10 = 0
    tau_obs = n11/m - n01/(n-m)
    M = np.zeros(n-N11+1, dtype=int)
    while (N10 <= (n-N11-N01)) and (N01 <= (n-N11)):
        pl = pval_one_lower(n, m, [N11, N10, N01, n-(N11+N10+N01)],
                            Z_all, tau_obs)
        if pl >= alpha:
            M[N01] = N10
            N01 += 1
        else:
            N10 += 1
    if N01 <= (n - N11):
        for i in np.arange(N01, (n-N11+1)):
            M[i] = n+1
    N11_vec0 = np.full((n-N11+1), N11)
    N10_vec0 = M
    N01_vec0 = np.arange(n-N11+1)
    N11_vec = np.empty(0, dtype=int)
    N10_vec = np.empty(0, dtype=int)
    N01_vec = np.empty(0, dtype=int)
    for i in np.arange(len(N11_vec0)):
        if N10_vec0[i] <= (n-N11_vec0[i]-N01_vec0[i]):
            N10_vec = np.append(N10_vec, np.arange(N10_vec0[i], n-N11_vec0[i] -
                                                   N01_vec0[i]+1))
            N11_vec = np.append(N11_vec, np.full((n-N11_vec0[i]-N01_vec0[i] -
                                                  N10_vec0[i]+1),
                                                 N11_vec0[i]))
            N01_vec = np.append(N01_vec, np.full((n-N11_vec0[i]-N01_vec0[i] -
                                                  N10_vec0[i]+1), N01_vec0[i]))
    compat = check_compatible(n11, n10, n01, n00, N11_vec, N10_vec, N01_vec)
    if sum(compat) > 0:
        tau_min = min(N10_vec[compat] - N01_vec[compat])/n
        accept_pos = np.flatnonzero(N10_vec[compat]-N01_vec[compat] ==
                                    n*tau_min)
        accept_pos = accept_pos[0]
        N_accept = np.array([N11, N10_vec[compat][accept_pos],
                             N01_vec[compat][accept_pos],
                             n-(N11+N10_vec[compat][accept_pos] +
                                N01_vec[compat][accept_pos])])
    else:
        tau_min = (n11 + n00)/n
        N_accept = float('NaN')
    return (tau_min, N_accept)


def tau_lower_oneside(n11, n10, n01, n00, alpha, nperm):
    """
    Calculate tau_lower, tau_upper, and N_accept for method I.

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
    alpha: float
        1 - confidence level
    nperm: int
        maximum desired number of permutations

    Returns
    -------
    tau_lower: float
        left-side tau for one-sided confidence interval
    tau_upper: float
        right-side tau for one-sided confidence interval
    N_accept: numpy array
        accepted potential table for one-sided confidence interval
    """
    n = n11+n10+n01+n00
    m = n11+n10
    if comb(n, m) <= nperm:
        Z_all = nchoosem(n, m)
    else:
        Z_all = combs(n, m, nperm)
    tau_min = (n11+n00)/n
    N_accept = float('NaN')
    for N11 in np.arange(n11+n01+1):
        tau_min_N11 = tau_lower_N11_oneside(n11, n10, n01, n00, N11, Z_all,
                                            alpha)
        if tau_min_N11[0] < tau_min:
            N_accept = tau_min_N11[1]
        tau_min = min(tau_min, tau_min_N11[0])
    tau_lower = tau_min
    tau_upper = (n11+n00)/n
    return (tau_lower, tau_upper, N_accept)


def tau_lower_N11_twoside(n11, n10, n01, n00, N11, Z_all, alpha):
    """
    Calculate tau_min and N_accept for method 3.

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
        number of subjects under control and treatment with potential outcome 1
    Z_all: numpy array
        re-randomization or sample of re-randomization matrix
    alpha: float
        1 - confidence level

    Returns
    -------
    tau_min: float
        minimum tau value of accepted potential tables
    tau_max: float
        minimum tau value of accepted potential tables
    N_accept_min: numpy array
        minimum accepted potential table
    N_accept_max: numpy array
        maximum accepted potential table
    rand_test_num: int
        number of tests run
    """
    n = n11 + n10 + n01 + n00
    m = n11 + n10
    tau_obs = n11/m - n01/(n-m)
    ntau_obs = n*n11/m - n*n01/(n-m)
    N10 = 0
    N01_vec0 = np.arange(n-N11+1)[np.arange(n-N11+1) >= -ntau_obs]
    N01 = min(N01_vec0)
    M = np.empty(len(N01_vec0), dtype=int)
    rand_test_num = 0
    while (N10 <= (n-N11-N01)) and (N01 <= (n-N11)):
        if N10 <= (N01+ntau_obs):
            pl = pval_two(n, m, [N11, N10, N01, n-(N11 + N10 + N01)], Z_all,
                          tau_obs)
            rand_test_num += 1
            if pl >= alpha:
                M[N01_vec0 == N01] = N10
                N01 += 1
            else:
                N10 += 1
        else:
            M[N01_vec0 == N01] = N10
            N01 += 1
    if N01 <= (n-N11):
        M[N01_vec0 >= N01] = np.floor(N01_vec0[N01_vec0 >= N01]+ntau_obs)+1
        N11_vec0 = [N11]*len(N01_vec0)
    N11_vec0 = np.full(len(N01_vec0), N11)
    N10_vec0 = M
    N11_vec = np.empty(0, dtype=int)
    N10_vec = np.empty(0, dtype=int)
    N01_vec = np.empty(0, dtype=int)
    for i in np.arange(len(N11_vec0)):
        N10_upper = int(min((n-N11_vec0[i]-N01_vec0[i]), np.floor(N01_vec0[i] +
                                                                  ntau_obs)))
        if N10_vec0[i] <= N10_upper:
            N10_vec = np.append(N10_vec, np.arange(N10_vec0[i], N10_upper+1))
            N11_vec = np.append(N11_vec, np.full((N10_upper-N10_vec0[i]+1),
                                                 N11_vec0[i]))
            N01_vec = np.append(N01_vec, np.full((N10_upper-N10_vec0[i]+1),
                                                 N01_vec0[i]))
    compat = check_compatible(n11, n10, n01, n00, N11_vec, N10_vec, N01_vec)
    if sum(compat) > 0:
        tau_min = min(N10_vec[compat] - N01_vec[compat])/n
        accept_pos = np.flatnonzero(N10_vec[compat]-N01_vec[compat] ==
                                    n*tau_min)
        accept_pos = accept_pos[0]
        N_accept_min = np.array([N11, N10_vec[compat][accept_pos],
                                 N01_vec[compat][accept_pos],
                                 n-(N11+N10_vec[compat][accept_pos] +
                                    N01_vec[compat][accept_pos])])

        tau_max = max(N10_vec[compat] - N01_vec[compat])/n
        accept_pos = np.flatnonzero(N10_vec[compat]-N01_vec[compat] ==
                                    n*tau_max)
        accept_pos = accept_pos[0]
        N_accept_max = np.array([N11, N10_vec[compat][accept_pos],
                                 N01_vec[compat][accept_pos],
                                 n-(N11+N10_vec[compat][accept_pos] +
                                    N01_vec[compat][accept_pos])])
    else:
        tau_min = np.inf
        N_accept_min = np.nan
        tau_max = np.NINF
        N_accept_max = np.nan
    return (tau_min, tau_max, N_accept_min, N_accept_max, rand_test_num)


def tau_twoside_lower(n11, n10, n01, n00, alpha, Z_all, exact, reps):
    """
    Calculate taus and N_accepts for method 3.

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
    alpha: float
        1 - confidence level
    Z_all: numpy array
        re-randomization or sample of re-randomization matrix
    exact: boolean
        whether or not to calculate exact confidence interval
    reps:
        if exact = False, number of simulations per table

    Returns
    -------
    tau_lower: float
        left-side tau for two-sided confidence interval
    N_accept_lower: numpy array
        left-side accepted potential table for two-sided confidence interval
    tau_upper: float
        right-side tau for two-sided confidence interval
    N_accept_upper: numpy array
        right-side accepted potential table for two-sided confidence interval
    rand_test_total: int
        number of tests run
    """
    n = n11+n10+n01+n00
    m = n11+n10
    tau_obs = n11/m - n01/(n-m)
    ntau_obs = n*n11/m - n*n01/(n-m)
    tau_min = np.inf
    tau_max = np.NINF
    N_accept_min = np.nan
    N_accept_max = np.nan
    rand_test_total = 0
    for N11 in np.arange(int(min(n11+n01, n+ntau_obs))+1):
        tau_min_N11 = tau_lower_N11_twoside(n11, n10, n01, n00, N11, Z_all,
                                            alpha)
        rand_test_total = rand_test_total + tau_min_N11[4]
        if tau_min_N11[0] < tau_min:
            N_accept_min = tau_min_N11[2]
        if tau_min_N11[1] > tau_max:
            N_accept_max = tau_min_N11[3]
        tau_min = min(tau_min, tau_min_N11[0])
        tau_max = max(tau_max, tau_min_N11[1])
        if (not exact) and (rand_test_total >= reps):
            break
    tau_lower = tau_min
    tau_upper = tau_max
    N_accept_lower = N_accept_min
    N_accept_upper = N_accept_max
    return (tau_lower, N_accept_lower, tau_upper, N_accept_upper,
            rand_test_total)


def tau_twoside_less_treated(n11, n10, n01, n00, alpha, exact,
                             max_combinations, reps):
    """
    Calculate taus and N_accepts for method 3.

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
    alpha: float
        1 - confidence level
    exact: boolean
        whether or not to calculate exact confidence interval
    max_combinations: int
        if exact = True, maximum desired number of combinations
    reps: int
        if exact = False, number of simulations per table

    Returns
    -------
    tau_lower: float
        left-side tau for two-sided confidence interval
    tau_upper: float
        right-side tau for two-sided confidence interval
    N_accept_lower: numpy array
        left-side accepted potential table for two-sided confidence interval
    N_accept_upper: numpy array
        right-side accepted potential table for two-sided confidence interval
    rand_test_total: int
        number of tests run
    """
    n = n11+n10+n01+n00
    m = n11+n10
    if exact:
        if comb(n, m) <= max_combinations:
            Z_all = nchoosem(n, m)
        else:
            raise Exception('Not enough combinations. Increase \
                             max_combinations to ' + str(comb(n, m)) +
                            ' for exact interval.')
    else:
        if comb(n, m) <= max_combinations:
            Z_all = nchoosem(n, m)
        else:
            Z_all = combs(n, m, reps)
    ci_lower = tau_twoside_lower(n11, n10, n01, n00, alpha, Z_all, exact, reps)
    ci_upper = tau_twoside_lower(n10, n11, n00, n01, alpha, Z_all, exact, reps)
    rand_test_total = ci_lower[4] + ci_upper[4]
    tau_lower = min(ci_lower[0], -1*ci_upper[2])
    tau_upper = max(ci_lower[2], -1*ci_upper[0])
    if tau_lower == ci_lower[0]:
        N_accept_lower = ci_lower[1]
    else:
        N_accept_lower = np.flip(ci_upper[3])
    if tau_upper == -1*ci_upper[0]:
        N_accept_upper = np.flip(ci_upper[1])
    else:
        N_accept_upper = ci_lower[3]
    return (tau_lower, tau_upper, N_accept_lower, N_accept_upper,
            rand_test_total)


def tau_twosided_ci(n11, n10, n01, n00, alpha, exact, max_combinations, reps):
    """
    Calculate taus and N_accepts for method 3.

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
    alpha: float
        1 - confidence level
    exact: boolean
        whether or not to calculate exact confidence interval
    max_combinations: int
        if exact = True, maximum desired number of combinations
    reps: int
        if exact = False, number of simulations per table

    Returns
    -------
    tau_lower: float
        left-side tau for two-sided confidence interval
    tau_upper: float
        right-side tau for two-sided confidence interval
    N_accept_lower: list
        left-side accepted potential table for two-sided confidence interval
    N_accept_upper: list
        right-side accepted potential table for two-sided confidence interval
    rand_test_total: int
        number of tests run
    """
    n = n11+n10+n01+n00
    m = n11+n10
    if m > (n/2):
        ci = tau_twoside_less_treated(n11, n10, n01, n00, alpha, exact,
                                      max_combinations, reps)
        tau_lower = -ci[1]
        tau_upper = -ci[0]
        N_accept_lower = ci[2]
        N_accept_upper = ci[3]
        rand_test_total = ci[4]
    else:
        ci = tau_twoside_less_treated(n11, n10, n01, n00, alpha, exact,
                                      max_combinations, reps)
        tau_lower = ci[0]
        tau_upper = ci[1]
        N_accept_lower = ci[2]
        N_accept_upper = ci[3]
        rand_test_total = ci[4]
    if exact:
        num_tables = len(nchoosem(n, m))
    else:
        num_tables = len(combs(n, m, reps))
    return ([int(tau_lower*n), int(tau_upper*n)],
            [N_accept_lower.tolist(), N_accept_upper.tolist()],
            [num_tables, rand_test_total])


def ind(x, a, b):
    """
    Indicator function for a <= x <= b.

    Parameters
    ----------
    x: int
        desired value
    a: int
        lower bound of interval
    b:
        upper bound of interval

    Returns
    -------
    1 if a <= x <= b and 0 otherwise.
    """
    return (x >= a)*(x <= b)


def lci(x, n, N, alpha):
    """
    Calculate lower confidence bound.

    Parameters
    ----------
    x: int/numpy array
        number(s) of good items in the sample
    n: int
        sample size
    N: int
        population size
    alpha: float
        1 - confidence level

    Returns
    -------
    kk: int/numpy array
        lower bound(s)
    """
    if isinstance(x, int):
        x = np.array([x])
    kk = np.arange(0, len(x))
    for i in kk:
        if x[i] < 0.5:
            kk[i] = 0
        else:
            aa = np.arange(0, N+1)
            bb = (aa + 1).astype(np.float64)
            bb[1:(N+1)] = hypergeom.cdf(x[i]-1, N, aa[1:(N+1)]-1, n)
            dd = np.vstack((aa, bb))
            dd = dd[:, dd[1] >= (1-alpha)]
            if dd.shape[0]*dd.shape[1] == 2:
                kk[i] = dd[0, 0]
            else:
                kk[i] = max(dd[0])
    if isinstance(x, int):
        return kk[0]
    else:
        return kk


def uci(x, n, N, alpha):
    """
    Calculate upper confidence bound.

    Parameters
    ----------
    x: int/numpy array
        number(s) of good items in the sample
    n: int
        sample size
    N: int
        population size
    alpha: float
        1 - confidence level

    Returns
    -------
    kk: int/numpy array
        upper bound(s)
    """
    if isinstance(x, int):
        xs = [x]
    else:
        xs = x
    lcis = lci(n-x, n, N, alpha)
    upper = N - lcis
    if isinstance(x, int):
        return upper[0]
    else:
        return upper


def exact_CI_odd(N, n, x, alpha):
    """
    Calculate exact CI for odd sample size.

    Parameters
    ----------
    N: int
        population size
    n: int (odd)
        sample size
    x: int
        number of good items in the sample
    alpha:
        1 - confidence level

    Returns
    -------
    lower: int
        lower bound of confidence interval
    upper: int
        upper bound of confidence interval
    """
    xx = np.arange(n+1)
    lcin1 = lci(xx, n, N, alpha/2)
    ucin1 = uci(xx, n, N, alpha/2)
    lcin2 = lci(xx, n, N, alpha)
    ucin2 = uci(xx, n, N, alpha)
    lciw = lcin1
    uciw = ucin1
    xvalue = int(floor(n/2)+1)
    while xvalue > -0.5:
        al = lcin2[xvalue]-lciw[xvalue]+1
        au = int(uciw[xvalue] - ucin2[xvalue]+1)
        if al*au > 1:
            ff = np.zeros((al*au, 4))
            for i in np.arange(al):
                ff[np.arange(i*au, i*au+au), 0] = lciw[xvalue]+i
                ff[np.arange(i*au, i*au+au), 1] = np.arange(ucin2[xvalue],
                                                            uciw[xvalue]+1)
                ff[np.arange(i*au, i*au+au), 2] = (
                    ff[np.arange(i*au, i*au+au), 1] -
                    ff[np.arange(i*au, i*au+au), 0])
            for ii in np.arange(len(ff)):
                lciw[xvalue] = ff[ii, 0]
                uciw[xvalue] = ff[ii, 1]
                lciw[n-xvalue] = N-uciw[xvalue]
                uciw[n-xvalue] = N-lciw[xvalue]

                def cpci(M):
                    kk = np.arange(len(M)).astype(np.float64)
                    for i in np.arange(len(M)):
                        xx = np.arange(n+1)
                        indp = xx.astype(np.float64)
                        uu = 0
                        while (uu < n + 0.5):
                            indp[uu] = (ind(M[i], lciw[uu], uciw[uu]) *
                                        hypergeom.pmf(uu, N, M[i], n))
                            uu += 1
                        kk[i] = sum(indp)
                    return kk
                M = np.arange(N+1)
                ff[ii, 3] = min(cpci(M))
            ff = ff[ff[:, 3] >= (1-alpha), :]
            if ff.shape[0]*ff.shape[1] > 4:
                ff = sorted(ff, key=lambda x: x[2])
                lciw[xvalue] = ff[0][0]
                uciw[xvalue] = ff[0][1]
            else:
                lciw[xvalue] = ff[0][0]
                uciw[xvalue] = ff[0][1]
            lciw[n-xvalue] = N - uciw[xvalue]
            uciw[n-xvalue] = N - lciw[xvalue]
        xvalue -= 1
    lower = lciw[xx == x]
    upper = uciw[xx == x]
    return (lower, upper)


def exact_CI_even(N, n, x, alpha):
    """
    Calculate exact CI for even sample size.

    Parameters
    ----------
    N: int
        population size
    n: int (even)
        sample size
    x: int
        number of good items in the sample
    alpha:
        1 - confidence level

    Returns
    -------
    lower: int
        lower bound of confidence interval
    upper: int
        upper bound of confidence interval
    """
    xx = np.arange(n+1)
    lcin1 = lci(xx, n, N, alpha/2)
    ucin1 = uci(xx, n, N, alpha/2)
    lcin2 = lci(xx, n, N, alpha)
    ucin2 = uci(xx, n, N, alpha)
    lciw = lcin1
    uciw = ucin1
    xvalue = int((n/2))
    aa = np.arange(lciw[xvalue], floor(N/2)+1)
    ii = 1
    while ii < (len(aa) + 0.5):
        lciw[xvalue] = aa[ii - 1]
        uciw[xvalue] = N - aa[ii - 1]

        def cpci(M):
            kk = np.arange(len(M)).astype(np.float64)
            for i in np.arange(len(M)):
                xx = np.arange(n+1)
                indp = xx.astype(np.float64)
                uu = 0
                while (uu < n + 0.5):
                    indp[uu] = (ind(M[i], lciw[uu], uciw[uu]) *
                                hypergeom.pmf(uu, N, M[i], n))
                    uu += 1
                kk[i] = sum(indp)
            return kk
        M = np.arange(N+1)
        bb = min(cpci(M))
        if (bb >= 1-alpha):
            ii1 = ii
            ii += 1
        else:
            ii = len(aa) + 1
    lciw[xvalue] = aa[ii1-1]
    uciw[xvalue] = N - lciw[xvalue]
    xvalue = int((n/2)-1)
    while xvalue > -0.5:
        al = lcin2[xvalue]-lciw[xvalue]+1
        au = int(uciw[xvalue]-ucin2[xvalue]+1)
        if al*au > 1:
            ff = np.zeros((al*au, 4))
            for i in np.arange(al):
                ff[np.arange(i*au, i*au+au), 0] = lciw[xvalue]+i
                ff[np.arange(i*au, i*au+au), 1] = np.arange(ucin2[xvalue],
                                                            uciw[xvalue]+1)
                ff[np.arange(i*au, i*au+au), 2] = (
                    ff[np.arange(i*au, i*au+au), 1] -
                    ff[np.arange(i*au, i*au+au), 0])
            for ii in np.arange(len(ff)):
                lciw[xvalue] = ff[ii, 0]
                uciw[xvalue] = ff[ii, 1]
                lciw[n-xvalue] = N-uciw[xvalue]
                uciw[n-xvalue] = N-lciw[xvalue]

                def cpci(M):
                    kk = np.arange(len(M)).astype(np.float64)
                    for i in np.arange(len(M)):
                        xx = np.arange(n+1)
                        indp = xx.astype(np.float64)
                        uu = 0
                        while (uu < n + 0.5):
                            indp[uu] = (ind(M[i], lciw[uu], uciw[uu]) *
                                        hypergeom.pmf(uu, N, M[i], n))
                            uu += 1
                        kk[i] = sum(indp)
                    return kk
                M = np.arange(N+1)
                ff[ii, 3] = min(cpci(M))
            ff = ff[ff[:, 3] >= (1-alpha), :]
            print(ff)
            if ff.shape[0]*ff.shape[1] > 4:
                ff = sorted(ff, key=lambda x: x[2])
                lciw[xvalue] = ff[0][0]
                uciw[xvalue] = ff[0][1]
            else:
                lciw[xvalue] = ff[0][0]
                uciw[xvalue] = ff[0][1]
            lciw[n-xvalue] = N - uciw[xvalue]
            uciw[n-xvalue] = N - lciw[xvalue]
        xvalue -= 1
    lower = lciw[xx == x]
    upper = uciw[xx == x]
    return (lower, upper)


def exact_CI(N, n, x, alpha):
    """
    Calculate exact CI for even or odd sample size.

    Parameters
    ----------
    N: int
        population size
    n: int (even)
        sample size
    x: int
        number of good items in the sample
    alpha:
        1 - confidence level

    Returns
    -------
    lower: int
        lower bound of confidence interval
    upper: int
        upper bound of confidence interval
    """
    if n % 2 == 1:
        ci = exact_CI_odd(N, n, x, alpha)
    else:
        ci = exact_CI_even(N, n, x, alpha)
    lower = int(ci[0])
    upper = int(ci[1])
    return (lower, upper)


def combin_exact_CI(n11, n10, n01, n00, alpha):
    """
    Calculate taus for method 2.1.

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
    alpha: float
        1 - confidence level

    Returns
    -------
    tau_lower: float
        left-side tau for one-sided confidence interval
    tau_upper: float
        right-side tau for one-sided confidence interval
    """
    n = n11+n10+n01+n00
    m = n11+n10
    ci_1plus = exact_CI(N=n, n=m, x=n11, alpha=alpha/2)
    ci_plus1 = exact_CI(N=n, n=(n-m), x=n01, alpha=alpha/2)
    tau_upper = (ci_1plus[1]-ci_plus1[0])/n
    tau_lower = (ci_1plus[0]-ci_plus1[1])/n
    return (tau_lower, tau_upper)


def N_plus1_exact_CI(n11, n10, n01, n00, alpha):
    """
    Calculate taus for method 2.2.

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
    alpha: float
        1 - confidence level

    Returns
    -------
    tau_lower: float
        left-side tau for one-sided confidence interval
    tau_upper: float
        right-side tau for one-sided confidence interval
    """
    n = n11+n10+n01+n00
    m = n11+n10
    tau_min = float('inf')
    tau_max = float('-inf')
    ci_plus1 = exact_CI(N=n, n=(n-m), x=n01, alpha=alpha)
    for N_plus1 in np.arange(ci_plus1[0], ci_plus1[1]+1):
        for N11 in np.arange(N_plus1+1):
            N01 = N_plus1-N11
            for N10 in np.arange(n-N_plus1+1):
                N00 = n-N_plus1-N10
                if check_compatible(n11, n10, n01, n00, np.array([N11]),
                                    np.array([N10]), np.array([N01]))[0]:
                    tau = (N10-N01)/n
                    tau_min = min(tau, tau_min)
                    tau_max = max(tau, tau_max)
    upper = tau_max
    lower = tau_min
    return (lower, upper)

print("hello")
