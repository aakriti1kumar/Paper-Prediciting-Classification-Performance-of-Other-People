import pickle
import numpy as np
import pandas as pd
import stan
from load_data import load_data
import predict_score

model_self = """data {
    int n_items;
    int K; 
    int<lower=1,upper=K> Y[n_items];
    real a_true;
    vector[n_items] d_true;
    vector[K-1] v;
  }
  parameters {
    real noise_a;
    real gamma;
    real lambda;
    real<lower=0> sigma;
    real noise_d;
    real<lower=0> sigma_d;
    real<lower=0> sigma_a;

  }
  transformed parameters{
    vector[n_items] d;
    real a; 
    a = a_true + noise_a;
    d = gamma*d_true + lambda + noise_d;
  }
  model {
    sigma_d ~ cauchy(0,2);
    sigma_a ~ cauchy(0,2);
    noise_a ~ normal(0, sigma_a);
    sigma ~ cauchy(0,2);
    gamma ~ std_normal();
    lambda ~ std_normal();
    noise_d ~ normal(0, sigma_d);
    for (j in 1:n_items){
        real p;
        vector[K] Pmf;
        p = inv_logit(a - d[j]);
        Pmf[1] = Phi((v[1] - p)/sigma);
        Pmf[K] = 1 - Phi((v[K-1] - p)/sigma);
        for (k in 2:(K-1)){ 
            Pmf[k] = Phi((v[k] - p)/sigma) - Phi((v[k-1] - p)/sigma);}
        Y[j] ~ categorical(Pmf);
  }   
 }
"""

df = pd.read_csv("Exp2_Estimation.csv")
data = load_data(df)
I = data['I']
J = data['J']
df_self = data['df_self']
v = data['v']

with open("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/true_params.pkl", "rb") as handle:
    true = pickle.load(handle)

d_true = np.zeros((I, J))
for i in range(I):
    d_true[i] = [true['d_true'][j] for j in (df_self['idSet'].values.reshape(128, 16)[i] - 1)]

a_true = true['a_true']
a = np.zeros(I)
lambda_ = np.zeros(I)
sigma = np.zeros(I)
sigma_a = np.zeros(I)
sigma_d = np.zeros(I)
gamma_ = np.zeros(I)
noise_a = np.zeros(I)
noise_d = np.zeros(I)
d = np.zeros((I, J))
Sim_SelfEst = np.zeros((I, J))
K = 13
for i in range(I):
    participant_data_self = {'n_items': data['J'], 'K': 13, 'Y': data['data_self_est'][i] + 1,
                             'a_true': a_true[i], 'd_true': d_true[i],
                             'v': v}

    posterior_self = stan.build(model_self, data=participant_data_self)
    fit_model_self = posterior_self.sample(num_chains=2, num_samples=1000)
    a[i] = fit_model_self['a'].mean()
    lambda_[i] = fit_model_self['lambda'].mean()
    gamma_[i] = fit_model_self['gamma'].mean()
    sigma[i] = fit_model_self['sigma'].mean()
    sigma_a[i] = fit_model_self['sigma_a'].mean()
    sigma_d[i] = fit_model_self['sigma_d'].mean()
    d[i] = fit_model_self['d'].mean(1)
    noise_d[i] = fit_model_self['noise_d'].mean()
    noise_a[i] = fit_model_self['noise_a'].mean()
    print(i)
    for j in np.arange(J):
        Sim_SelfEst[i, j] = predict_score.predict_score_hyp2(a[i], d[i, j], sigma[i], v, K=13)

self_params = {'a_self': a, 'd_self': d, 'gamma_self': gamma_, 'lambda_self': lambda_, 'sigma_self': sigma,
               'Sim_Y': Sim_SelfEst, 'noise_a': noise_a, 'noise_d': noise_d, 'sigma_a': sigma_a,
               'sigma_d': sigma_d}
with open("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/self_params.pkl", "wb") as tf:
    pickle.dump(self_params, tf)
