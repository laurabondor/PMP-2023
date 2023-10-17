from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx


model = BayesianNetwork([('C', 'I'), ('I', 'A'), ('C', 'A')])

cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.9995], [0.0005]])
cpd_i = TabularCPD(variable='I', variable_card=2, 
                   values=[[0.99, 0.03],
                           [0.01, 0.97]], 
                   evidence=['C'], 
                   evidence_card=[2])

cpd_a = TabularCPD(variable='A', variable_card=2, 
                   values=[[0.9998, 0.02, 0.02, 0.95], 
                           [0.0002, 0.98, 0.98, 0.05]],
                   evidence=['C', 'I'],
                   evidence_card=[2, 2])

model.add_cpds(cpd_c, cpd_i, cpd_a)

assert model.check_model()

# 2. Ştiind că alarma de incendiu a fost declanşată, calculaţi probabilitatea să fi avut loc un cutremur
infer = VariableElimination(model)
result1 = infer.query(variables=['C'], evidence={'A': 1})
print(result1)

# 3. Afişaţi probabilitatea ca un incendiu sa fi avut loc, fără ca alarma de incendiu să se activeze
result2 = infer.query(variables=['I'], evidence={'A': 0})
print(result2)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
#plt.show()

nume_fisier = "grafic.png"
plt.savefig(nume_fisier)