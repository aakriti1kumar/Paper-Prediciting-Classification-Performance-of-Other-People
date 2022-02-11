from open_file import open_file
import pickle
import numpy as np

hyp1_25 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo4/hyp2-feedback/hyp2_feedback-25.pkl")
hyp1_50 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo4/hyp2-feedback/hyp2_feedback-50.pkl")
hyp1_64 = open_file("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo4/hyp2-feedback/hyp2_feedback-64.pkl")

a_other_full = hyp1_25['a_other'] + hyp1_50['a_other'] + hyp1_64['a_other']

d_other_full = hyp1_25['d_other'] + hyp1_50['d_other'] + hyp1_64['d_other']

Sim_OtherEst_full = (hyp1_25['Sim_OtherEst']) + (hyp1_50['Sim_OtherEst']) + (hyp1_64['Sim_OtherEst'])

hyp1_final = {'a_other': a_other_full, 'd_other': d_other_full, 'Sim_OtherEst': Sim_OtherEst_full}

with open('/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo4/hyp2-feedback/hyp2_feedback64.pkl', "wb") as tf:
    pickle.dump(hyp1_final, tf)
