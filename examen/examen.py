import pandas as pd
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom

#sub 1
#a 
data = pd.read_csv('Titanic.csv')
mean_age = data['Age'].mean()
data['Age'].fillna(mean_age, inplace=True)
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
print(data.head())

#b
with pm.Model() as model:
    beta_age = pm.Normal('beta_age', mu=0, tau=1/10**2)
    beta_class = pm.Normal('beta_class', mu=0, tau=1/10**2)
    alpha = pm.Normal('alpha', mu=0, tau=1/10**2)
    mu = alpha + beta_age * data['Age'] + beta_class * data['Pclass']
    p_survive = pm.invlogit(mu)
    survived_obs = pm.Bernoulli('survived_obs', p=p_survive, observed=data['Survived'].values)

with model:
    trace = pm.sample(1000, tune=1000)

pm.summary(trace)

#c
# Variabilele care influenteaza rezultatul sunt atat beta_age cat si beta_class

#d
p_survive_posterior = pm.invlogit(trace['alpha'] + trace['beta_age'] * 30 + trace['beta_class'] * 2)
hdi_90 = pm.utils.hpd(p_survive_posterior, alpha=0.1)

print(f"Intervalul de 90% HDI: {hdi_90}")

#sub 2
param_X = 0.3
param_Y = 0.5
N = 10000
k = 30

results = []

for i in range(k):
    x = geom.rvs(param_X, size=N)
    y = geom.rvs(param_Y, size=N)
    inside = x > y**2
    prob = inside.sum() / N
    results.append(prob)

mean_result = np.mean(results)
std_dev_result = np.std(results)

print(f'Aproximarea medie pentru P(X > Y^2) dupa {k} iteratii: {mean_result:.4f}')
print(f'Deviatia standard: {std_dev_result:.4f}')

plt.hist(results, bins=20, density=True, alpha=0.7)
plt.xlabel('P(X > Y^2)')
plt.show()