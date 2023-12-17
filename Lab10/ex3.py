import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az

az.style.use('arviz-darkgrid')

dummy_data = np.loadtxt('dummy.csv')
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]

order_cubic = 3

x_p_cubic = np.vstack([x_1**i for i in range(1, order+1)])
x_s_cubic = (x_p_cubic - x_p_cubic.mean(axis=1, keepdims=True)) / x_p_cubic.std(axis=1, keepdims=True)
y_s = (y_1 - y_1.mean()) / y_1.std()

with pm.Model() as model_cubic_500_points:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=10, shape=order_cubic)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_s_cubic)
    y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_s)
idata_cubic_500_points = pm.sample(2000, return_inferencedata=True, model=model_cubic_500_points)

waic_linear = az.waic(idata_l)
waic_quadratic = az.waic(idata_p)
waic_cubic = az.waic(idata_cubic_500_points)

loo_linear = az.loo(idata_l)
loo_quadratic = az.loo(idata_p)
loo_cubic = az.loo(idata_cubic_500_points)

x_plot = np.linspace(x_s[0].min(), x_s[0].max(), 100)
x_plot_cubic = np.vstack([x_plot**i for i in range(1, order_cubic+1)])
x_plot_s_cubic = (x_plot_cubic - x_plot_cubic.mean(axis=1, keepdims=True)) / x_plot_cubic.std(axis=1, keepdims=True)

α_cubic_post = idata_cubic_500_points.posterior['α'].mean(("chain", "draw")).values
β_cubic_post = idata_cubic_500_points.posterior['β'].mean(("chain", "draw")).values
y_cubic_post = α_cubic_post
for i in range(1, order_cubic + 1):
    y_cubic_post += β_cubic_post[i - 1] * x_plot ** i

plt.plot(x_plot, y_cubic_post, 'C4', label='model cubic')
plt.plot(x_new, y_p_order_5_500_post, 'C3', label=f'model order {order}')
plt.scatter(x_s[0], y_s, c='C0', marker='.')
plt.legend()
plt.show()

print("WAIC - Linear:", waic_linear)
print("WAIC - Quadratic:", waic_quadratic)
print("WAIC - Cubic:", waic_cubic)

print("LOO - Linear:", loo_linear)
print("LOO - Quadratic:", loo_quadratic)
print("LOO - Cubic:", loo_cubic)