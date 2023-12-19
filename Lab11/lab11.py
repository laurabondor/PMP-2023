import numpy as np
import arviz as az
import pymc3 as pm
import matplotlib.pyplot as plt

np.random.seed(123)
clusters = 3
n_cluster = [200, 150, 100]
n_total = sum(n_cluster)
means = [5, 0, -5]
std_devs = [2, 2, 2]
data = np.concatenate([np.random.normal(mu, sigma, size=n) for mu, sigma, n in zip(means, std_devs, n_cluster)])

az.plot_kde(data)
plt.show()

with pm.Model() as model_2:
    p = pm.Dirichlet('p', a=np.ones(2))
    means = pm.Normal('means', mu=np.linspace(data.min(), data.max(), 2), sigma=2, shape=2, transform=pm.distributions.transforms.ordered)
    sd = pm.HalfNormal('sd', sigma=2)
    y = pm.NormalMixture('y', w=p, mu=means, sigma=sd, observed=data)
    idata_2 = pm.sample(1000, tune=2000, target_accept=0.9, random_seed=123, return_inferencedata=True)

with pm.Model() as model_3:
    p = pm.Dirichlet('p', a=np.ones(3))
    means = pm.Normal('means', mu=np.linspace(data.min(), data.max(), 3), sigma=2, shape=3, transform=pm.distributions.transforms.ordered)
    sd = pm.HalfNormal('sd', sigma=2)
    y = pm.NormalMixture('y', w=p, mu=means, sigma=sd, observed=data)
    idata_3 = pm.sample(1000, tune=2000, target_accept=0.9, random_seed=123, return_inferencedata=True)

with pm.Model() as model_4:
    p = pm.Dirichlet('p', a=np.ones(4))
    means = pm.Normal('means', mu=np.linspace(data.min(), data.max(), 4), sigma=2, shape=4, transform=pm.distributions.transforms.ordered)
    sd = pm.HalfNormal('sd', sigma=2)
    y = pm.NormalMixture('y', w=p, mu=means, sigma=sd, observed=data)
    idata_4 = pm.sample(1000, tune=2000, target_accept=0.9, random_seed=123, return_inferencedata=True)

waic_scores = []
loo_scores = []
models = [model_2, model_3, model_4]

for model in models:
    with model:
        idata = pm.sample(1000, tune=2000, random_seed=123, return_inferencedata=True)
        waic = az.waic(idata)
        loo = az.loo(idata)
        waic_scores.append(waic)
        loo_scores.append(loo)

for i, (waic, loo) in enumerate(zip(waic_scores, loo_scores), 2):
    print(f"Model cu {i} componente:")
    print(f"WAIC: {waic.waic}")
    print(f"LOO: {loo.loo}\n")