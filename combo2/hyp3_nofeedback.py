import numpy as np
import pandas as pd
import pickle
from load_data import load_data
from open_file import open_file
import predict_score

self = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/self_params.pkl")
df = pd.read_csv("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/Exp2_Estimation.csv")
df_nofeedback = df[df.conditionshowFeedback == 0]
data = load_data(df_nofeedback)
I = data['I']
J = data['J']
df_self = data['df_self']
v = data['v']
idx = data['idx']


a_other = self['a_self'][idx]
d_other = self['d_self'][idx]
sigma = self['sigma_self'][idx]
Sim_OtherEst = np.zeros((I, J))
for i in range(64):
    for j in range(J):
        Sim_OtherEst[i, j] = predict_score.predict_score_hyp2(a_other[i], d_other[i, j], sigma[i], v, K=13)
    print(Sim_OtherEst[i])


    other_hyp3 = {'Sim_OtherEst': Sim_OtherEst}

with open("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo2/hyp3-nofeedback/hyp3_nofeedback.pkl", "wb") as tf:
    pickle.dump(other_hyp3, tf)
