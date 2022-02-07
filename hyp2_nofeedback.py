import numpy as np
import pandas as pd
import pickle
from scipy.special import expit
from scipy.stats import norm
from scipy import stats
from load_data import load_data
from open_file import open_file
import predict_score

true = open_file("true_params.pkl")
self = open_file("self_params.pkl")

df = pd.read_csv("Exp2_Estimation.csv")
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
delta = np.zeros((I))
a_other = np.zeros((I))
K = 13
# Pmf = np.zeros(K)
for i in range(I):
    delta[i] = np.random.uniform(-1,1,1)
    a_other[i] = a_self[i] + delta[i]
    for j in range(J):
        Sim_OtherEst[i, j] = predict_score.predict_score_hyp2(a_other[i], d_other[i, j], sigma[i], v, K=13)


other_hyp2 = {'a_other': a_other, 'delta': delta, 'Sim_OtherEst': Sim_OtherEst}

with open("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/hyp2-nofeedback/hyp2_nofeedback.pkl", "wb") as tf:
    pickle.dump(other_hyp2, tf)
