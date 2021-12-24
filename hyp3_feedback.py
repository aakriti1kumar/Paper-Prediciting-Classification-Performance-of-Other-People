import numpy as np
import pandas as pd
import pickle
import nest_asyncio
nest_asyncio.apply()
from scipy.special import expit
from scipy.stats import norm
from scipy import stats

with open("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/true_params.pkl", "rb") as handle:
    true = pickle.load(handle)

with open("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/self_params.pkl", "rb") as handle:
    self = pickle.load(handle)

df = pd.read_csv("Exp2_Estimation.csv")
df_feedback = df[df.conditionshowFeedback == 1]
data = load_data(df_feedback)
I = data['I']
J = data['J']
df_self = data['df_self']
v = data['v']
data_other_true = data['data_other_true']
idx = data['idx']


a_other = self['a_self'][idx]
d_other = self['d_self'][idx]
sigma = self['sigma_self'][idx]
Sim_OtherEst = np.zeros((I, J))
K = 13
Pmf = np.zeros(K)
for i in range(64):
    for j in range(J):
        p = expit(a_other[i] - d_other[i,j])
        Pmf[0] = norm.cdf((v[0] - p)/sigma[i])
        Pmf[(K-1)] = 1 - norm.cdf((v[11] - p)/sigma[i])
        for k in range(K-2):
            Pmf[k+1] = norm.cdf((v[k+1] - p)/sigma[i]) - norm.cdf((v[k] - p)/sigma[i])
        custm = stats.rv_discrete(name='custm', values=(np.linspace(0,12,13),Pmf))
        Sim_OtherEst[i, j] = custm.rvs(size=1)

other_hyp3 = {'Sim_OtherEst': Sim_OtherEst}

with open("hyp3_feedback.pkl", "wb") as tf:
    pickle.dump(other_hyp3, tf)
