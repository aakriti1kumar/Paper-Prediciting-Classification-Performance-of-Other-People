import pickle
import numpy as np
import nest_asyncio
import pandas as pd
import stan
nest_asyncio.apply()
from load_data import load_data
import predict_score
from scipy.stats import norm
from scipy import stats
from scipy.special import expit

with open("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/true_params.pkl", "rb") as handle:
    true = pickle.load(handle)

with open("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/self_params.pkl", "rb") as handle:
    self = pickle.load(handle)

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

for i in np.arange(50,64):
    a_other[i,0] = np.random.randn(1)
    d_other[i,0] = np.random.randn(1)
    Pmf = np.zeros(K)
    p = expit(a_other[i,0] - d_other[i,0])
    Pmf[0] = norm.cdf((v[0] - p)/sigma[i])
    Pmf[(K-1)] = 1 - norm.cdf((v[11] - p)/sigma[i])
    for k in range(K-2):
        Pmf[k+1] = norm.cdf((v[k+1] - p)/sigma[i]) - norm.cdf((v[k] - p)/sigma[i])
    custm = stats.rv_discrete(name='custm', values=(np.linspace(0,12,13),Pmf))
    Sim_OtherEst_hyp1[i,0] = custm.rvs(size=1)
    for j in np.arange(1,J):
        participant_data_other_hyp1 = {'n_items': j, 'K': 12+1, 'Y': data_other_true[i,:j] +1,
                            'v': v, 'sigma': sigma[i]}

        posterior_other_hyp1 = stan.build(model_hyp1, data=participant_data_other_hyp1)
        fit_model_other = posterior_other_hyp1.sample(num_chains=2, num_samples=1000)
        a_other[i,j] = fit_model_other['a_other'].mean()
        d_other[i,j] = fit_model_other['d_other'].mean()
        mu_d[i,j] = fit_model_other['mu_d'].mean()
        sigma_d[i,j] = fit_model_other['sigma_d'].mean()
        Sim_OtherEst_hyp1[i,j] = predict_score(a_other[i,j], mu_d[i,j], sigma_d[i,j], sigma[i], v, K=13)
        print((i)*j)
other_hyp2 = {'a_other': a_other, 'd_other': d_other, 'Sim_OtherEst': np.around(Sim_OtherEst)}

with open("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/hyp2_feedback.pkl", "wb") as tf:
    pickle.dump(other_hyp2, tf)
