import pickle
import numpy as np
import pandas as pd
from load_data import load_data
import predict_score
from open_file import open_file

self = open_file("self_params.pkl")
df = pd.read_csv("Exp2_Estimation.csv")
df_feedback = df[df.conditionshowFeedback == 0]
data = load_data(df_feedback)
I = data['I']
J = data['J']
df_self = data['df_self']
v = data['v']
idx = data['idx']


a = self['a_self'][idx]
d = self['d_self'][idx]
sigma = self['sigma_self'][idx]
Sim_SelfEst = np.zeros((I, J))


for i in np.arange(I):
    for j in np.arange(J):
        Sim_SelfEst[i, j] = predict_score.predict_score_hyp2(a[i], d[i, j], sigma[i], v, K=13)

self_params = {'Sim_Y': Sim_SelfEst}
with open("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/self_sim_nf.pkl", "wb") as tf:
    pickle.dump(self_params, tf)
