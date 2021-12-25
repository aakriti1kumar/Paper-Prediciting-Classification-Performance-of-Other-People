import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
import matplotlib.patches as mpatches
import set_style

set_style.set_style()

with open("true_params.pkl", "rb") as handle:
    true = pickle.load(handle)

with open("self_params.pkl", "rb") as handle:
    self = pickle.load(handle)

with open("hyp1_feedback.pkl", "rb") as handle:
    hyp1 = pickle.load(handle)

with open("hyp2_feedback.pkl", "rb") as handle:
    hyp2 = pickle.load(handle)

with open("hyp3_feedback.pkl", "rb") as handle:
    hyp3 = pickle.load(handle)


