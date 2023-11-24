#sub1
#1
import random

def aruncare_moneda(masluita=False):
    if masluita:
        return 1 if random.random() < 2/3 else 0
    return random.randint(0, 1)

def joc():
    jucator_initial = random.choice(['J0', 'J1'])
    steme_j0 = aruncare_moneda()
    steme_j1 = aruncare_moneda(True) if jucator_initial == 'J1' else aruncare_moneda()

    if steme_j0 >= steme_j1:
        return jucator_initial
    return 'J1' if jucator_initial == 'J0' else 'J0'

nr_jocuri = 10000
castig_j0 = sum(joc() == 'J0' for _ in range(nr_jocuri))
castig_j1 = nr_jocuri - castig_j0

print("j0: ", castig_j0)
print("j1: ", castig_j1)

#2
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

model = BayesianNetwork([('J0', 'J1'), ('J1', 'castig')])

#Definirea variabilei radacina
j0 = TabularCPD(variable='J0', variable_card=2, values=[[0.5], [0.5]])

#Definirea variabilor cu un parinte
j1 = TabularCPD(variable='J1', variable_card=3, values=[[1/3, 2/3], [1/3, 1/3], [2/3, 1/3]],
                evidence=['J0'], evidence_card=[2])

#Definirea variabilor cu doi parinti 
castig = TabularCPD(variable='castig', variable_card=2, values=[[1, 0, 0, 1], [0, 1, 1, 0]], 
                    evidence=['J0', 'J1'], evidence_card=[2, 3])

#Adaugarea distributiilor conditionale la model
model.add_cpds(j0, j1, castig)

#Verificarea modelului
print(model.check_model())

#3
#Inferenta
inferenta = VariableElimination(model)

prob_j0 = inferenta.query(variables=['J0'], evidence={'castig': 1})

print(prob_j0)


#sub2
#1
import numpy as np

np.random.seed(42) 
mu = 10  
sigma = 2  
timp_mediu_asteptare = np.random.normal(mu, sigma, 100)

print(timp_mediu_asteptare)

#2
import pymc3 as pm
import matplotlib.pyplot as plt

with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=5)

    timp_mediu_asteptare = pm.Normal('timp_mediu_asteptare', mu=mu, sd=sigma, observed=timp_mediu_asteptare_obs)

with model:
    trace = pm.sample(2000, tune=1000)

pm.traceplot(trace)
plt.show()

#3
pm.plot_posterior(trace['mu'], credible_interval=0.95)
plt.xlabel('mu')
plt.ylabel('Density')
plt.title('Distributia a posteriori pentru mu')
plt.show()