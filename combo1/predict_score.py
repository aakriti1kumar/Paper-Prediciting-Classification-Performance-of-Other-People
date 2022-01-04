from scipy.stats import norm
from scipy import stats
from scipy.special import expit
import numpy as np
def predict_score_hyp1(a, mu_d, sigma_d, sigma, v, K=13):
    Pmf = np.zeros(K)
    d = np.random.normal(mu_d, sigma_d, 1)
    p = expit(a - d)
    Pmf[0] = norm.cdf((v[0] - p) / sigma)
    Pmf[(K - 1)] = 1 - norm.cdf((v[11] - p) / sigma)
    for k in range(K - 2):
        Pmf[k + 1] = norm.cdf((v[k + 1] - p) / sigma) - norm.cdf((v[k] - p) / sigma)
    custm = stats.rv_discrete(name='custm', values=(np.linspace(0, 12, 13), Pmf))
    return custm.rvs(size=100).mean()


def predict_score_hyp2(a, d, sigma, v, K=13):
    Pmf = np.zeros(K)
    p = expit(a - d)
    Pmf[0] = norm.cdf((v[0] - p) / sigma)
    Pmf[(K - 1)] = 1 - norm.cdf((v[11] - p) / sigma)
    for k in range(K - 2):
        Pmf[k + 1] = norm.cdf((v[k + 1] - p) / sigma) - norm.cdf((v[k] - p) / sigma)
    custm = stats.rv_discrete(name='custm', values=(np.linspace(0, 12, 13), Pmf))
    return custm.rvs(size=100).mean()
