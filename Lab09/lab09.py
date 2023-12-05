import pymc3 as pm
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

data = pd.read_csv('Admission.csv')

with pm.Model() as logistic_model:
    beta0 = pm.Normal('beta0', mu=0, sd=10)
    beta1 = pm.Normal('beta1', mu=0, sd=10)
    beta2 = pm.Normal('beta2', mu=0, sd=10)

    p = 1 / (1 + np.exp(-(beta0 + beta1 * data['GRE'] + beta2 * data['GPA'])))
    admit = pm.Bernoulli('admit', p=p, observed=data['Admission'])
    trace = pm.sample(2000, tune=1000, cores=-1)

az.plot_trace(trace)
plt.show()

beta0_samples = trace['beta0']
beta1_samples = trace['beta1']
beta2_samples = trace['beta2']

decision_boundary = np.mean(-(beta0_samples + beta1_samples * data['GRE'] + beta2_samples * data['GPA']))

print("Granita de decizie: ", decision_boundary)

hdi_94 = az.hdi(p, hdi_prob=0.94)
print("Intervalul HDI de 94%: ", hdi_94)

plt.scatter(data['GRE'], data['GPA'], c=data['Admission'], cmap='viridis', label='Data')
plt.xlabel('GRE')
plt.ylabel('GPA')

x_vals = np.linspace(data['GRE'].min(), data['GRE'].max(), 100)
y_vals = (-beta0_samples.mean() - beta1_samples.mean() * x_vals) / beta2_samples.mean()
plt.plot(x_vals, y_vals, label='Granita de decizie', color='red')

plt.legend()
plt.colorbar(label='Admitere')
plt.show()

prob_student_1 = 1 / (1 + np.exp(-(beta0_samples + beta1_samples * 550 + beta2_samples * 3.5)))
hdi_90_student_1 = az.hdi(prob_student_1, hdi_prob=0.9)
print("Intervalul HDI de 90% pt cerinta 3:", hdi_90_student_1)

prob_student_2 = 1 / (1 + np.exp(-(beta0_samples + beta1_samples * 500 + beta2_samples * 3.2)))
hdi_90_student_2 = az.hdi(prob_student_2, hdi_prob=0.9)
print("Intervalul HDI de 90% pt cerinta 4:", hdi_90_student_2)