import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
alpha = 3  
waiting_times = []

for i in range(100):
    
    poisson_samples = np.random.poisson(20)

    wait_times = np.random.normal(2, 0.5, poisson_samples)
   
    avg_wait_time = np.mean(wait_times)
    waiting_times.append(avg_wait_time)


with pm.Model() as model:
    alpha_estimated = pm.Exponential("alpha", 1 / alpha)
    observed_data = pm.Normal("observed_data", mu=2, sigma=0.5, observed=waiting_times)
    
    trace = pm.sample(10000, tune=1000, chains=2)


plt.figure(figsize=(8, 6))
pm.plots.plot_posterior(trace["alpha"], kde_plot=True)
plt.title("Distributia estimata a lui alfa")

nume_fisier = "grafic.png"
plt.savefig(nume_fisier)

print(pm.summary(trace))