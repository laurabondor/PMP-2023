import pymc3 as pm
import pandas as pd

data = pd.read_csv('trafic.csv')

trafic_observed = data['nr. masini'].values

ore_modificare = [7, 8, 16, 19]

with pm.Model() as model:
    
    lambda_ = pm.Exponential("lambda", 1.0)

    trafic_mediu = pm.Poisson("trafic_mediu", lambda_, shape=len(trafic_observed))

    delta_lambda = pm.Normal("delta_lambda", mu=0, sigma=1, shape=len(ore_modificare))

    for i, hour in enumerate(ore_modificare):
        trafic_mediu = pm.math.switch(pm.math.gt(range(len(trafic_observed)), hour * 60), trafic_mediu + delta_lambda[i], trafic_mediu)

    trafic_observat = pm.Poisson("trafic_observat", trafic_mediu, observed=trafic_observed)

    trace = pm.sample(1000, tune=1000)

pm.summary(trace)

