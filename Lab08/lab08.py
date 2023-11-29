import pandas as pd
import pymc3 as pm
import numpy as np

#1
data = pd.read_csv('Prices.csv')

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)
    sigma = pm.HalfCauchy('sigma', beta=5)

    mu = alpha + beta1 * data['Speed'] + beta2 * np.log(data['HardDrive'])
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=data['Price'])
    trace = pm.sample(2000, tune=1000)

#2
hdi_beta1 = pm.stats.hdi(trace['beta1'], hdi_prob=0.95)
hdi_beta2 = pm.stats.hdi(trace['beta2'], hdi_prob=0.95)
print("Beta1:", hdi_beta1)
print("Beta2:", hdi_beta2)

#3
if (hdi_beta1[0] < 0 < hdi_beta1[1]) or (hdi_beta2[0] < 0 < hdi_beta2[1]):
    print("Nu sunt predictori utili")
else:
    print("Sunt predictori utili")

#4
specificatii = {'Speed': 33, 'HardDrive': 540}
with model:
    pred_samples = pm.sample_posterior_predictive(trace, samples=5000, vars=[y_obs], new_obs=specificatii)

hdi_pred_price = pm.stats.hdi(pred_samples['y_obs'], hdi_prob=0.9)
print("Intervalul de 90% HDI pt cerinta 4: ", hdi_pred_price)


#5
with model:
    post_pred = pm.sample_posterior_predictive(trace, samples=5000)

hdi_prediction = pm.stats.hdi(post_pred['y_obs'], hdi_prob=0.9)
print("Intervalul de 90% HDI pt cerinta 5: ", hdi_prediction)
