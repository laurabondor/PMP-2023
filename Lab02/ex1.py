import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az

lambda1 = 4  
lambda2 = 6  

x = np.random.choice([1, 2], size=10000, p=[0.4, 1 - 0.4])
X = np.where(x == 1, stats.expon.rvs(scale=1 / lambda1), stats.expon.rvs(scale=1 / lambda2))

media_X = np.mean(x)
deviatia_X = np.std(x)

print(media_X)
print(deviatia_X)

az.plot_posterior({'x':x}) 

#plt.show()
nume_fisier = "grafic.png"
plt.savefig(nume_fisier)