from pgmpy.models import BayesianNetwork
from pgmpy.factors.continuous import ContinuousFactor
from pgmpy.inference import VariableElimination


model = BayesianNetwork([('X', 'Y'), ('X', 'Z')])
  
media = 2  
deviatia = 0.5    

cpd_X = ContinuousFactor(variables=['X'], pdf=lambda x: 1 / (20**x) * np.exp(-20) / math.factorial(x)) 
cpd_Y = ContinuousFactor(variables=['Y'], pdf=lambda y: (1 / (deviatia * math.sqrt(2 * math.pi))) * np.exp(-(y - media)**2 / (2 * deviatia**2)) 
cpd_Z = ContinuousFactor(variables=['Z'], pdf=lambda z: 1.0 * np.exp(-1.0 * z)) 

model.add_cpds(cpd_X, cpd_Y, cpd_Z)

#voi incarca pana la sfârșitul zilei și restul exercițiilor 

