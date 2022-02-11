import pickle
import nest_asyncio
import pandas as pd
import stan
nest_asyncio.apply()
from load_data import load_data


# Stan Model
model_true = """data {
    int<lower=0> n_participants;
    int<lower=0> n_items;
    int<lower=0> K; //max score possible on each item
    int<lower=1,upper=K> Y[n_participants,n_items];
    vector[K-1] v;
  }
  parameters {
    vector[n_participants] a; //ability
    vector[n_items] d; //difficulty
    real<lower=0> mu_d; //mu for d
    real<lower=0> sigma_d; //sigma for d
    real<lower=0> sigma; //sigma for p_prime
  }
  model {
    a ~ normal(0,1);
    d ~ normal(mu_d,sigma_d);
    mu_d ~ normal(0,2);
    sigma_d ~ cauchy(0,2);
    sigma ~ cauchy(0,2);
    for(i in 1:n_participants){
      for (j in 1:n_items){
        real p;
        vector[K] Pmf;
        p = inv_logit(a[i] - d[j]);
        Pmf[1] = Phi((v[1] - p)/sigma);
        Pmf[K] = 1 - Phi((v[K-1] - p)/sigma);
        for (k in 2:(K-1)){ 
            Pmf[k] = Phi((v[k] - p)/sigma) - Phi((v[k-1] - p)/sigma);}
        Y[i,j] ~ categorical(Pmf);
     } 
    }   
  }  
"""

df = pd.read_csv("Exp2_Estimation.csv")
data = load_data(df)

participant_true_data = dict(n_participants=data['I'],
                             n_items=data['J'],
                             K=12 + 1,
                             Y=data['data_self_true'] + 1,
                             v=data['v'])
posterior_true = stan.build(model_true, data=participant_true_data)
fit_model_true = posterior_true.sample(num_chains=2, num_samples=1000)

a_true = fit_model_true['a'].mean(axis=1)
d_true = fit_model_true['d'].mean(axis=1)
sigma = fit_model_true['sigma'].mean(axis=1)
mu_d = fit_model_true['mu_d'].mean(axis=1)
sigma_d = fit_model_true['sigma_d'].mean(axis=1)

true_params = dict(a_true=a_true, d_true=d_true, mu_d=mu_d, sigma_d=sigma_d, sigma=sigma)
with open("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/true_params.pkl", "wb") as tf:
    pickle.dump(true_params, tf)
