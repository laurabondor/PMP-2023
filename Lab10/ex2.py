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

x_p = np.vstack([x_1**i for i in range(1, order+1)])
x_s = (x_p - x_p.mean(axis=1, keepdims=True)) / x_p.std(axis=1, keepdims=True)
y_s = (y_1 - y_1.mean()) / y_1.std()

with pm.Model() as model_p_order_5_500_points:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=10, shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_s)
    y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_s)
idata_p_order_5_500_points = pm.sample(2000, return_inferencedata=True, model=model_p_order_5_500_points)

x_plot = np.linspace(x_s[0].min(), x_s[0].max(), 100)
α_p_order_5_500_post = idata_p_order_5_500_points.posterior['α'].mean(("chain", "draw")).values
β_p_order_5_500_post = idata_p_order_5_500_points.posterior['β'].mean(("chain", "draw")).values

y_p_order_5_500_post = α_p_order_5_500_post
for i in range(1, order + 1):
    y_p_order_5_500_post += β_p_order_5_500_post[i - 1] * x_plot ** i

plt.plot(x_plot, y_p_order_5_500_post, 'C3', label=f'model order {order}')
plt.scatter(x_s[0], y_s, c='C0', marker='.')
plt.legend()
plt.show()