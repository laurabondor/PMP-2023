import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#1
def posterior_grid(grid_points=50, heads=6, tails=9, prior_type='uniform'):
    grid = np.linspace(0, 1, grid_points)
    
    if prior_type == 'uniform':
        prior = np.repeat(1/grid_points, grid_points)  
    elif prior_type == 'binary':
        prior = (grid <= 0.5).astype(int)  
    elif prior_type == 'absolute_difference':
        prior = abs(grid - 0.5)  
    else:
        prior = np.ones(grid_points) 
    
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

data = np.repeat([0, 1], (100, 30))  
points = 50  
h = data.sum()
t = len(data) - h
prior_types = ['uniform', 'binary', 'absolute_difference']  

plt.figure(figsize=(15, 5))
for i, prior_type in enumerate(prior_types):
    grid, posterior = posterior_grid(points, h, t, prior_type)
    plt.subplot(1, len(prior_types), i + 1)
    plt.plot(grid, posterior, 'o-')
    plt.title(f'Prior Type: {prior_type}\nheads = {h}, tails = {t}')
    plt.yticks([])
    plt.xlabel('θ')

plt.tight_layout()
plt.show()

#2
def estimate_pi(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum() * 4 / N
    error = abs((pi - np.pi) / np.pi) * 100
    return error

Ns = [100, 1000, 10000]
num_simulations = 100  
errors = np.zeros((len(Ns), num_simulations))

for i, N in enumerate(Ns):
    for j in range(num_simulations):
        errors[i, j] = estimate_pi(N)

mean_errors = np.mean(errors, axis=1)
std_errors = np.std(errors, axis=1)

plt.errorbar(Ns, mean_errors, yerr=std_errors, fmt='o', capsize=5)
plt.xscale('log')
plt.xlabel('Number of Points (N)')
plt.ylabel('Error (%)')
plt.title('Error Estimation for π vs. Number of Points')
plt.show()

#3
def metropolis(func, draws=10000):
    trace = np.zeros(draws)
    old_x = 0.5  
    old_prob = func.pdf(old_x)
    delta = np.random.normal(0, 0.5, draws)
    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = func.pdf(new_x)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    return trace

a_params = [1, 20, 1]  
b_params = [1, 20, 4]  

n_trials = [0, 1, 2, 3, 4, 8, 16, 32, 50, 150]
data = [0, 1, 1, 1, 1, 4, 6, 9, 13, 48]

plt.figure(figsize=(10, 8))
dist = stats.beta
x = np.linspace(0, 1, 200)

for idx, N in enumerate(n_trials):
    if idx == 0:
        plt.subplot(4, 3, 2)
        plt.xlabel('θ')
    else:
        plt.subplot(4, 3, idx+3)
        plt.xticks([])
    y = data[idx]
    for (a_prior, b_prior) in zip(a_params, b_params):
        func = stats.beta(a_prior + y, b_prior + N - y) 
        metropolis_trace = metropolis(func)  
        plt.fill_between(x, 0, func.pdf(x), alpha=0.7)  
        plt.hist(metropolis_trace, bins=50, density=True, alpha=0.5)  
    plt.axvline(0.35, ymax=0.3, color='black')
    plt.plot(0, 0, label=f'{N:4d} aruncari\n{y:4d} steme', alpha=0)
    plt.xlim(0, 1)
    plt.ylim(0, 12)
    plt.legend()
    plt.yticks([])
plt.tight_layout()
plt.show()