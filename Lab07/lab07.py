import pymc3 as pm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("auto-mpg.csv")
data = data[data['horsepower'].notna()]
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')

sns.scatterplot(x='horsepower', y='mpg', data=data)
plt.title('Relatia dintre CP si mpg')
plt.xlabel('CP')
plt.ylabel('mpg')
#plt.show()
plt.savefig("grafic1.png")

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)
    mu = alpha + beta * data['horsepower']
    sigma = pm.HalfNormal('sigma', sd=10)
    mpg = pm.Normal('mpg', mu=mu, sd=sigma, observed=data['mpg'])
    
with model:
    trace = pm.sample(2000, tune=1000)

alpha_pred = trace['alpha'].mean()
beta_pred = trace['beta'].mean()

sns.scatterplot(x='horsepower', y='mpg', data=data)
plt.plot(data['horsepower'], alpha_pred + beta_pred * data['horsepower'], color='red', label='Dreapta de regresie')
plt.title('Regresie liniara')
plt.xlabel('CP')
plt.ylabel('mpg')
plt.legend()
#plt.show()
plt.savefig("grafic2.png")

with model:
    pm.plot_posterior_predictive_glm(trace, samples=100, eval=np.linspace(data['horsepower'].min(), data['horsepower'].max(), 100), label='Distributia predictiva a posteriori')
    plt.title('Distributia predictiva a posteriori')
    plt.xlabel('CP')
    plt.ylabel('mpg')
    plt.legend()
    #plt.show()
    plt.savefig("grafic3.png")