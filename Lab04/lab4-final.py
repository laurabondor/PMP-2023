from pgmpy.models import BayesianNetwork
from pgmpy.factors.continuous import ContinuousFactor
from pgmpy.inference import VariableElimination
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import networkx as nx
import math
import numpy as np

model = BayesianNetwork([('X', 'Y'), ('X', 'Z')])

media = 2
deviatia = 0.5

cpd_X = ContinuousFactor(variables=['X'], pdf=lambda x: 1 / (20**x) * np.exp(-20) / math.factorial(x))
cpd_Y = ContinuousFactor(variables=['Y'], pdf=lambda y: (1 / (deviatia * math.sqrt(2 * math.pi))) * np.exp(-(y - media)**2 / (2 * deviatia**2)))
cpd_Z = ContinuousFactor(variables=['Z'], pdf=lambda z: 1.0 * np.exp(-1.0 * z))

model.add_cpds(cpd_X, cpd_Y, cpd_Z)

assert model.check_model()

infer = VariableElimination(model)

def objective(alpha):
    result = infer.query(variables=['X'], evidence={'Y': 15 - alpha, 'Z': 15 - alpha})
    return 1 - result.values[0]

alpha = 0.0  

result = minimize(objective, alpha, bounds=[(0, None)])
alpha_max = result.x[0]

result = infer.query(variables=['Y'], evidence={'X': 1, 'Z': alpha_max})
average_wait_time = result.values[0]

print(alpha_max)
print(average_wait_time)