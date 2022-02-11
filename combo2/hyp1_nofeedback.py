import numpy as np
import pandas as pd
import pickle
from load_data import load_data
import predict_score

with open("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/true_params.pkl", "rb") as handle:
    true = pickle.load(handle)

with open("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/self_params.pkl", "rb") as handle:
    self = pickle.load(handle)

df = pd.read_csv("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/Exp2_Estimation.csv")
df_feedback = df[df.conditionshowFeedback == 0]
data = load_data(df_feedback)
I = data['I']
J = data['J']
df_self = data['df_self']
v = data['v']
data_other_true = data['data_other_true']
idx = data['idx']

a_other = np.random.randn(I)
d_other = np.random.randn((I * J)).reshape(I, J)
sigma = self['sigma_self'][idx]
Sim_OtherEst = np.zeros((I, J))
K = 13
Pmf = np.zeros(K)
for i in range(I):
    for j in range(J):
        Sim_OtherEst[i, j] = predict_score.predict_score_hyp2(a_other[i], d_other[i, j], sigma[i], v,
                                                              K=13)

hyp1 = {'Sim_OtherEst': Sim_OtherEst}
with open("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo2/hyp1-nofeedback/hyp1_nofeedback.pkl", "wb") as tf:
    pickle.dump(hyp1, tf)
