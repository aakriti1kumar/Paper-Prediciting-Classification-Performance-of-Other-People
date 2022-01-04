import pickle
import numpy as np
import pandas as pd
import stan
from load_data import load_data
from scipy.stats import norm
from scipy import stats
from scipy.special import expit
from open_file import open_file
import predict_score

true = open_file("true_params.pkl")
self = open_file("self_params.pkl")

df = pd.read_csv("Exp2_Estimation.csv")
df_feedback = df[df.conditionshowFeedback == 1]
data = load_data(df_feedback)
I = data['I']
J = data['J']
df_self = data['df_self']
v = data['v']
idx = data['idx']
data_other_true = data['data_other_true']

model_hyp2 = """data {
    int n_items;
    int<lower=0> K; //max score possible on each item
    int<lower=1,upper=K> Y[n_items];
    real d_other[n_items+1];
    real a_self;
    vector[K-1] v;
    real<lower=0> sigma;
  }
  parameters{
    real delta;
    real mu_delta_current;
    real<lower=0> sigma_delta_current;
  }
  transformed parameters{
    real a_other;
    a_other = a_self + delta;
  }
  model {
    delta ~ normal(mu_delta_current,sigma_delta_current);
    mu_delta_current ~ normal(0,1);
    sigma_delta_current ~ cauchy(0,2);
    for (j in 1:n_items){
        real p;
        p = inv_logit(a_other - d_other[j]);
        vector[K] Pmf;
        Pmf[1] = Phi((v[1] - p)/sigma);
        Pmf[K] = 1 - Phi((v[K-1] - p)/sigma);
        for (k in 2:(K-1)){ 
            Pmf[k] = Phi((v[k] - p)/sigma) - Phi((v[k-1] - p)/sigma);}
        Y[j] ~ categorical(Pmf);  }
 } 
  """

a_self = self['a_self'][idx]
d_other = self['d_self'][idx]
sigma = self['sigma_self'][idx]
Sim_OtherEst = np.zeros((I, J))
delta = np.zeros((I, J))
a_other = np.zeros((I, J))
K = 13

K = 13

for i in np.arange(50, 64):
    a_other[i, 0] = a_self[i] + np.random.randn(1)
    Pmf = np.zeros(K)
    p = expit(a_other[i, 0] - d_other[i, 0])
    Pmf[0] = norm.cdf((v[0] - p) / sigma[i])
    Pmf[(K - 1)] = 1 - norm.cdf((v[11] - p) / sigma[i])
    for k in range(K - 2):
        Pmf[k + 1] = norm.cdf((v[k + 1] - p) / sigma[i]) - norm.cdf((v[k] - p) / sigma[i])
    custm = stats.rv_discrete(name='custm', values=(np.linspace(0, 12, 13), Pmf))
    Sim_OtherEst[i, 0] = custm.rvs(size=1)
    for j in np.arange(1, J):
        participant_data_other_hyp2 = {'n_items': j, 'K': 12 + 1, 'Y': data_other_true[i, :j] + 1,
                                       'a_self': a_self[i], 'd_other': d_other[i, :(j + 1)],
                                       'v': v, 'sigma': sigma[i]}

        posterior_hyp2 = stan.build(model_hyp2, data=participant_data_other_hyp2)
        fit_model_other = posterior_hyp2.sample(num_chains=2, num_samples=1000)
        a_other[i, j] = fit_model_other['a_other'].mean()
        delta[i, j] = fit_model_other['delta'].mean()
        Sim_OtherEst[i, j] = predict_score.predict_score_hyp2(a_other[i, j], d_other[i, j], sigma[i], v, K=13)
        print(i,j)
other_hyp2 = {'a_other': a_other, 'd_other': d_other, 'Sim_OtherEst': np.around(Sim_OtherEst)}

with open('/Users/aakritikumar/Desktop/Lab/ToM-pycharm/hyp2-feedback/hyp2_feedback-64.pkl', "wb") as tf:
    pickle.dump(other_hyp2, tf)
