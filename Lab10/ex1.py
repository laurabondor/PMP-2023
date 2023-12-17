import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az

az.style.use('arviz-darkgrid')

dummy_data = np.loadtxt('dummy.csv')
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]

order = 5

x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)

with pm.Model() as model_p_order_5:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=10, shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
idata_p_order_5 = pm.sample(2000, return_inferencedata=True, model=model_p_order_5)

x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
α_p_order_5_post = idata_p_order_5.posterior['α'].mean(("chain", "draw")).values
β_p_order_5_post = idata_p_order_5.posterior['β'].mean(("chain", "draw")).values

y_p_order_5_post = α_p_order_5_post
for i in range(1, order + 1):
    y_p_order_5_post += β_p_order_5_post[i - 1] * x_new ** i

plt.plot(x_new, y_p_order_5_post, 'C3', label=f'model order {order}')
plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()
plt.show()

with pm.Model() as model_p_order_5_sd_100:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=100, shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
idata_p_order_5_sd_100 = pm.sample(2000, return_inferencedata=True, model=model_p_order_5_sd_100)

with pm.Model() as model_p_order_5_sd_array:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=[10, 0.1, 0.1, 0.1, 0.1])
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
idata_p_order_5_sd_array = pm.sample(2000, return_inferencedata=True, model=model_p_order_5_sd_array)