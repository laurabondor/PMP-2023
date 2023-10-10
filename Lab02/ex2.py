import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

alpha = [4, 4, 5, 5]
lambda_ = [1/3, 1/2, 1/2, 1/3] 
latency = 4  
prob_server = [0.25, 0.25, 0.30, 0.20]

server = []

for i in range(4):
    server.append(stats.gamma.rvs(alpha[i], scale=1/lambda_[i]))

samples_latency = stats.expon.rvs(scale=1/latency)

x = []
for i in range(10000):
    server_index = np.random.choice(4, p=prob_server)
    total_time = server[server_index][i] + samples_latency[i]
    x.append(total_time)

prob_final = np.mean(np.array(x) > 3)

print(prob_final)

plt.hist(x, bins=50, density=True, alpha=0.6)
#plt.show()

nume_fisier = "grafic.png"
plt.savefig(nume_fisier)