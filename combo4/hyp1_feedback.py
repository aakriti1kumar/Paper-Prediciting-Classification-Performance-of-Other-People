import pickle
import numpy as np
import pandas as pd
import stan
from load_data import load_data
import predict_score
from scipy.stats import norm
from scipy import stats
import combo_data
from scipy.special import expit
import shuffle_data


def run_hyp1(start, end):
    with open("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/self_params.pkl", "rb") as handle:
        self = pickle.load(handle)

    df = pd.read_csv("/Users/aakritikumar/Desktop/Lab/ToM-pycharm/Exp2_Estimation.csv")
    df_feedback = df[df.conditionshowFeedback == 1]
    data = load_data(df_feedback)
    I = data['I']
    J = data['J']
    v = data['v']
    idx = data['idx']
    data_other_true = data['data_other_true']
    data_self_idset = data['data_self_idset']
    other_id = combo_data.combo1_other_id
    data_other_true_shuffled = shuffle_data.shuffle(other_id, data_other_true, I, J)
    data_other_true_shuffled_idset = shuffle_data.shuffle_idset(data_self_idset, data_other_true_shuffled, I, J)


    model_hyp1 = """data {
        int n_items;
        int<lower=0> K; //max score possible on each item
        int<lower=1,upper=K> Y[n_items];
        vector[K-1] v;
        real<lower=0> sigma;
      }
      parameters{
        real a_other;
        vector[n_items] d_other;
        real mu_d;
        real<lower=0> sigma_d;
      }
      model {
        a_other ~ normal(0,1);
        d_other ~ normal(mu_d, sigma_d);
        mu_d ~ normal(0,2);
        sigma_d ~ cauchy(0,4);
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

    sigma = self['sigma_self'][idx]
    Sim_OtherEst = np.zeros((I, J))
    a_other = np.zeros((I, J))
    d_other = np.zeros((I, J))
    mu_d = np.zeros((I, J))
    sigma_d = np.zeros((I, J))
    K = 13
    for i in np.arange(start, end):
        a_other[i, 0] = np.random.randn(1)
        d_other[i, 0] = np.random.randn(1)
        Pmf = np.zeros(K)
        p = expit(a_other[i, 0] - d_other[i, 0])
        Pmf[0] = norm.cdf((v[0] - p) / sigma[i])
        Pmf[(K - 1)] = 1 - norm.cdf((v[11] - p) / sigma[i])
        for k in range(K - 2):
            Pmf[k + 1] = norm.cdf((v[k + 1] - p) / sigma[i]) - norm.cdf((v[k] - p) / sigma[i])
        custm = stats.rv_discrete(name='custm', values=(np.linspace(0, 12, 13), Pmf))
        Sim_OtherEst[i, 0] = custm.rvs(size=1)
        for j in np.arange(1, J):
            participant_data_other_hyp1 = {'n_items': j, 'K': 12 + 1, 'Y': data_other_true_shuffled_idset[i, :j] + 1,
                                           'v': v, 'sigma': sigma[i]}

            posterior_other_hyp1 = stan.build(model_hyp1, data=participant_data_other_hyp1)
            fit_model_other = posterior_other_hyp1.sample(num_chains=2, num_samples=1000)
            a_other[i, j] = fit_model_other['a_other'].mean()
            d_other[i, j] = fit_model_other['d_other'].mean()
            mu_d[i, j] = fit_model_other['mu_d'].mean()
            sigma_d[i, j] = fit_model_other['sigma_d'].mean()
            Sim_OtherEst[i, j] = predict_score.predict_score_hyp1(a_other[i, j], mu_d[i, j], sigma_d[i, j], sigma[i], v,
                                                                  K=13)
            print(i, j)
    other_hyp1 = {'a_other': a_other, 'd_other': d_other, 'Sim_OtherEst': Sim_OtherEst}

    with open(f"/Users/aakritikumar/Desktop/Lab/ToM-pycharm/combo4/hyp1-feedback/hyp1_feedback-{end}.pkl", "wb") as tf:
        pickle.dump(other_hyp1, tf)
    return None
