import numpy as np
import pandas as pd
import pickle
from scipy.special import expit
from scipy.stats import norm
from scipy import stats
from load_data import load_data
from open_file import open_file

true = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/true_params.pkl")
self = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/self_params.pkl")

df = pd.read_csv("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/Exp2_Estimation.csv")
df_feedback = df[df.conditionshowFeedback == 0]
data = load_data(df_feedback)
I = data['I']
J = data['J']
df_self = data['df_self']
v = data['v']
data_other_true = data['data_other_true']
idx = data['idx']

a_self = self['a_self'][idx]
d_other = self['d_self'][idx]
sigma = self['sigma_self'][idx]
Sim_OtherEst = np.zeros((I, J))
delta = np.random.randn((I))
a_other = np.zeros((I))
K = 13
Pmf = np.zeros(K)
for i in range(I):
    delta[i] = np.random.randn(1)
    a_other[i] = a_self[i] + delta[i]
    for j in range(J):
        p = expit(a_other[i] - d_other[i, j])
        Pmf[0] = norm.cdf((v[0] - p) / sigma[i])
        Pmf[(K - 1)] = 1 - norm.cdf((v[11] - p) / sigma[i])
        for k in range(K - 2):
            Pmf[k + 1] = norm.cdf((v[k + 1] - p) / sigma[i]) - norm.cdf((v[k] - p) / sigma[i])
        custm = stats.rv_discrete(name='custm', values=(np.linspace(0, 12, 13), Pmf))
        Sim_OtherEst[i, j] = custm.rvs(size=1)

other_hyp2 = {'a_other': a_other, 'delta': delta, 'Sim_OtherEst': Sim_OtherEst}

with open("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo1/hyp2-nofeedback/hyp2_nofeedback.pkl", "wb") as tf:
    pickle.dump(other_hyp2, tf)