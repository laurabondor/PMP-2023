import pymc3 as pm
import arviz as az

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

model = pm.Model()

with model:
    Y = pm.Categorical('Y', p=[1/len(Y_values)] * len(Y_values))
    theta = pm.Categorical('theta', p=[1/len(theta_values)] * len(theta_values))

with model:
    n = pm.Poisson('n', mu=10)
    p = theta * n
    observed = pm.Binomial('observed', n=n, p=p, observed=Y_values)

with model:
    trace = pm.sample(1000, tune=1000)

az.plot_posterior(trace)