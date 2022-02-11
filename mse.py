import numpy as np
import pandas as pd
from load_data import load_data
import pickle

df = pd.read_csv("Exp2_Estimation.csv")
df_feedback = df[df.conditionshowFeedback == 1]
data = load_data(df_feedback)

true = data['data_other_est']
true_true = data['data_other_true']
with open("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/hyp1-feedback/hyp1_feedback64.pkl", "rb") as handle:
    est_hyp1 = pickle.load(handle)


with open("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/hyp2-feedback/hyp2_feedback64.pkl", "rb") as handle:
    est_hyp2 = pickle.load(handle)


with open("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/hyp3-feedback/hyp3_feedback.pkl", "rb") as handle:
    est_hyp3 = pickle.load(handle)


print(f"MSE of hyp1 = {np.square(true - est_hyp1['Sim_OtherEst']).mean()}")

print(f"MSE of hyp2 = {np.square(true - est_hyp2['Sim_OtherEst']).mean()}")

print(f"MSE of hyp3 = {np.square(true - est_hyp3['Sim_OtherEst']).mean()}")



#
# print(f"MSE of hyp1 (w/o trial 1) = {np.square(true[:,1:] - est_hyp1['Sim_OtherEst'][:,1:]).mean()}")
#
# print(f"MSE of hyp2 (w/o trial 1) = {np.square(true[:,1:] - est_hyp2['Sim_OtherEst'][:,1:]).mean()}")
#
# print(f"MSE of hyp3 (w/o trial 1) = {np.square(true[:,1:] - est_hyp3['Sim_OtherEst'][:,1:]).mean()}")
#
#
#
# print(f"MSE of hyp1 (w/o trial 1 of each block) = {np.square(np.delete(true, np.s_[0,4,8,12], 1) - np.delete(est_hyp1['Sim_OtherEst'], np.s_[0,4,8,12], 1)).mean()}")
#
# print(f"MSE of hyp2 (w/o trial 1 of each block) = {np.square(np.delete(true, np.s_[0,4,8,12], 1) - np.delete(est_hyp2['Sim_OtherEst'], np.s_[0,4,8,12], 1)).mean()}")
#
# print(f"MSE of hyp3 (w/o trial 1 of each block) = {np.square(np.delete(true, np.s_[0,4,8,12], 1) - np.delete(est_hyp3['Sim_OtherEst'], np.s_[0,4,8,12], 1)).mean()}")



print(f"MSE true (w/o trial 1 of each block) = {np.square(np.delete(true, np.s_[0,4,8,12], 1) - np.delete(true_true, np.s_[0,4,8,12], 1)).mean()}")

print(f"MSE true (w/o trial 1) = {np.square(np.delete(true, np.s_[0], 1) - np.delete(true_true, np.s_[0], 1)).mean()}")

print(f"MSE true = {np.square(true- true_true).mean()}")