import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, binom


def bmle_discrete(Dm0, Mmin, delta):
    """
    MATLAB:
    function [b_estimation_MLE]=bmle_discrete(Dm0,Mmin,delta)
        b_estimation_MLE=1/(log(10)*(mean(Dm0)-Mmin+delta));
    end
    """
    return 1.0 / (np.log(10.0) * (np.mean(Dm0) - Mmin + delta))


def MBS_MLE_discrete(Dm, delta):
    """
    MATLAB:
    function [Mc,b_es,number]=MBS_MLE_discrete(Dm,delta)
    """
    dm = 0.1
    D00 = np.sort(Dm)[::-1]  # descending

    # MATLAB: m=ceil(min(D00)/dm)*dm:dm:floor(D00(50)/dm)*dm;
    # MATLAB indexing starts at 1, so D00(50) -> Python D00[49]
    m_start = np.ceil(np.min(D00) / dm) * dm
    m_end = np.floor(D00[49] / dm) * dm
    m = np.arange(m_start, m_end + 0.5 * dm, dm)

    b0 = np.zeros(len(m))
    sigma = np.zeros(len(m))

    for k in range(len(m)):
        Mmin = m[k]
        jkf = Dm >= (Mmin - 1e-11)
        Dm0 = Dm[jkf]
        b0[k] = bmle_discrete(Dm0, Mmin, delta)
        sigma[k] = b0[k] / np.sqrt(len(Dm0))

    Delta = 0.5
    number = int(Delta / dm + 1)

    b0_mean = np.array([
        np.mean(b0[k:k + number])
        for k in range(len(m) - number + 1)
    ])

    b02 = b0[:len(m) - number + 1]
    threshold = np.abs(b02 - b0_mean) - sigma[:len(m) - number + 1]

    jkf = np.where(threshold < 0)[0]

    if len(jkf) > 0:
        Mc = m[jkf[0]]
        jkff = np.where(Dm >= (Mc - 1e-11))[0]
        number_out = len(jkff)
        b_es = b02[jkf[0]]
    else:
        ind = np.argmin(threshold)
        Mc = m[ind]
        jkff = np.where(Dm >= (Mc - 1e-11))[0]
        number_out = len(jkff)
        b_es = b02[ind]

    return Mc, b_es, number_out


def Estimation_GR_discrete(Dm, Mmin, delta):
    """
    MATLAB:
    function [b_GR,loglikelihood_GR]=Estimation_GR_discrete(Dm,Mmin,delta)
    """
    DMoment = 10.0 ** (1.5 * (Dm + 6.07))
    Moment_min0 = 10.0 ** (1.5 * (Mmin + 6.07))
    Moment_min = 10.0 ** (1.5 * ((Mmin - delta) + 6.07))

    jkf = np.where((DMoment - Moment_min0) >= -1e-16)[0]
    n = len(jkf)

    beta_GR = n / np.sum(np.log(DMoment[jkf] / Moment_min))
    b_GR = beta_GR * 3.0 / 2.0

    loglikelihood_GR = (
        n * (beta_GR * np.log(Moment_min) + np.log(beta_GR))
        - (1.0 + beta_GR) * np.sum(np.log(DMoment[jkf]))
    )

    return b_GR, loglikelihood_GR


def Estimation_TGR_gridsearch_discrete(Dm, Mmin, delta, b_GR):
    """
    MATLAB:
    function [b_TGR_output,Mcorner_TGR_output,loglikelihood_TGR_output]=Estimation_TGR_gridsearch_discrete(Dm,Mmin,delta,b_GR)
    """
    DMoment = 10.0 ** (1.5 * (Dm + 6.07))
    Moment_min0 = 10.0 ** (1.5 * (Mmin + 6.07))
    Moment_min = 10.0 ** (1.5 * ((Mmin - delta) + 6.07))

    jkf = np.where((DMoment - Moment_min0) >= -1e-16)[0]
    n = len(jkf)

    Mcorner_TGR0 = np.arange(Mmin, np.ceil(np.max(Dm)) + 1 + 0.025, 0.05)
    Moment_corner_TGR0 = 10.0 ** (1.5 * (Mcorner_TGR0 + 6.07))

    b_TGR0 = np.arange(0.6, 1.5 + 0.025, 0.05)
    if b_GR >= 1.5:
        b_TGR0 = np.arange(1.3, 1.9 + 0.025, 0.05)
    elif b_GR <= 0.8:
        b_TGR0 = np.arange(0.4, 1.0 + 0.025, 0.05)

    # MATLAB initializes with 1e9 and then uses max(...), which is unusual,
    # but I preserve the actual behavior of taking the maximum.
    loglikelihood_TGR = np.full((len(b_TGR0), len(Moment_corner_TGR0)), 1e9)

    for b_num in range(len(b_TGR0)):
        b_TGR = b_TGR0[b_num]
        beta_TGR = 2.0 / 3.0 * b_TGR

        for Mcorner_TGR_num in range(len(Moment_corner_TGR0)):
            Moment_corner_TGR = Moment_corner_TGR0[Mcorner_TGR_num]

            loglikelihood_TGR[b_num, Mcorner_TGR_num] = (
                n * beta_TGR * np.log(Moment_min)
                + (1.0 / Moment_corner_TGR) * (n * Moment_min - np.sum(DMoment[jkf]))
                - beta_TGR * np.sum(np.log(DMoment[jkf]))
                + np.sum(np.log(beta_TGR / DMoment[jkf] + 1.0 / Moment_corner_TGR))
            )

    linear_index = np.argmax(loglikelihood_TGR)
    row, col = np.unravel_index(linear_index, loglikelihood_TGR.shape)

    b_TGR_output = b_TGR0[row]
    Mcorner_TGR_output = Mcorner_TGR0[col]
    loglikelihood_TGR_output = loglikelihood_TGR[row, col]

    return b_TGR_output, Mcorner_TGR_output, loglikelihood_TGR_output


def GR_confidential_interval(ax, b, N, mc_set):
    """
    MATLAB:
    function [pl] = GR_confidential_interval(b, N, mc_set)
    """
    color1 = np.array([[132, 94, 194],
                       [178, 91, 0],
                       [0, 139, 200]]) / 255.0

    n0 = np.arange(1, N + 1)
    dm = 0.01
    m_range = np.arange(0, 3 * np.log10(N) / b + 0.5 * dm, dm)
    P_large = 10.0 ** (-b * m_range)
    alpha = 90

    meadian = np.zeros(len(n0))
    small = np.zeros(len(n0))
    large = np.zeros(len(n0))

    for j in range(len(n0)):
        n = n0[j]
        P = binom.pmf(n, N, P_large)

        h0 = np.argmax(P)
        meadian[j] = m_range[h0]

        TT = np.cumsum(P) / np.sum(P)

        h1 = np.argmin(np.abs(TT - (1 - alpha / 100.0) / 2.0))
        h2 = np.argmin(np.abs(TT - (1 + alpha / 100.0) / 2.0))

        small[j] = m_range[h1]
        large[j] = m_range[h2]

    x_fill = np.concatenate([small + mc_set, (large + mc_set)[::-1]])
    y_fill = np.concatenate([n0, n0[::-1]])

    ax.fill(x_fill, y_fill, color=color1[2], alpha=0.1, edgecolor='none')
    pl, = ax.plot(meadian + mc_set, n0, '-', color=[0.2, 0.4, 0.9], linewidth=3)

    return pl


def calculate_TGR_GR_param(M0):
    idx = np.argsort(M0)[::-1]
    M0 = M0[idx]
    # T0 = T0[idx]

    # Magnitude processing
    diffm = np.diff(M0)
    nonzero_elements = diffm[diffm != 0]
    delta = np.min(np.abs(nonzero_elements)) / 2
    dm = 0.1

    # MLE estimate
    m = np.arange(np.floor(np.min(M0) / dm) * dm, np.max(M0) + dm, dm)

    # MATLAB hist(M0, m) uses m as bin centers
    # To mimic that, construct bin edges from centers
    edges = np.concatenate((
        [m[0] - dm / 2],
        (m[:-1] + m[1:]) / 2,
        [m[-1] + dm / 2]
    ))
    n0, _ = np.histogram(M0, bins=edges)

    ind = np.argmax(n0)
    Mc_MLE = m[ind] + 0.2

    # Mc_MBS, b_es, number = util_MFD.MBS_MLE_discrete(M0, delta)
    # mc = max(Mc_MLE, Mc_MBS, 1.5)
    mc = 1.5

    idx = (M0 - mc) >= -1e-11

    # Sort Dt and apply same order to Dm
    # Dt = np.sort(T0[idx])
    Dm = M0[idx] - mc

    mmin = 0
    b_GR, loglikelihood_GR = Estimation_GR_discrete(Dm, mmin, delta)
    b_TGR, mcorner_TGR, loglikelihood_TGR = Estimation_TGR_gridsearch_discrete(Dm, mmin, delta, b_GR)

    R = loglikelihood_TGR - loglikelihood_GR
    P_LR = 1 - chi2.cdf(2 * R, 1)
    
    
    return M0, Dm, mc, b_GR, b_TGR, mcorner_TGR, P_LR


def calculate_TGR_param(M0):
    # Sort M0 from high to low, and apply same index to T0
    idx = np.argsort(M0)[::-1]
    M0 = M0[idx]
    # T0 = T0[idx]

    # Magnitude processing
    diffm = np.diff(M0)
    nonzero_elements = diffm[diffm != 0]
    delta = np.min(np.abs(nonzero_elements)) / 2
    dm = 0.1

    # MLE estimate
    m = np.arange(np.floor(np.min(M0) / dm) * dm, np.max(M0) + dm, dm)

    # MATLAB hist(M0, m) uses m as bin centers
    # To mimic that, construct bin edges from centers
    edges = np.concatenate((
        [m[0] - dm / 2],
        (m[:-1] + m[1:]) / 2,
        [m[-1] + dm / 2]
    ))
    n0, _ = np.histogram(M0, bins=edges)

    ind = np.argmax(n0)
    Mc_MLE = m[ind] + 0.2

    # Mc_MBS, b_es, number = util_MFD.MBS_MLE_discrete(M0, delta)
    # mc = max(Mc_MLE, Mc_MBS, 1.5)
    mc = 1.5

    idx = (M0 - mc) >= -1e-11

    # Sort Dt and apply same order to Dm
    # Dt = np.sort(T0[idx])
    Dm = M0[idx] - mc

    mmin = 0
    b_GR, loglikelihood_GR = Estimation_GR_discrete(Dm, mmin, delta)
    b_TGR, mcorner_TGR, loglikelihood_TGR = Estimation_TGR_gridsearch_discrete(Dm, mmin, delta, b_GR)
    mcorner_TGR = mcorner_TGR + mc

    R = loglikelihood_TGR - loglikelihood_GR
    P_LR = 1 - chi2.cdf(2 * R, 1)
    
    
    return b_TGR, mcorner_TGR, P_LR



