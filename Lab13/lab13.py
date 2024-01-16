import arviz as az

centered_data = az.load_arviz_data("centered_eight") # modelul centrat
non_centered_data = az.load_arviz_data("non_centered_eight") # modelul necentrat

# 1
print("Model centrat:")
print("Nr de lanturi:", centered_data.posterior.chain.size)
print("Marimea totala a esantionului:", centered_data.posterior.shape[0])
az.plot_posterior(centered_data)
plt.show()

print("\nModel necentrat:")
print("Nr de lanturi:", non_centered_data.posterior.chain.size)
print("Marimea totală a esantionului:", non_centered_data.posterior.shape[0])
az.plot_posterior(non_centered_data)
plt.show()

# 2
summary_centered = az.summary(centered_data, var_names=["mu", "tau"], extend=False)
print(summary_centered["rhat"])

summary_non_centered = az.summary(non_centered_data, var_names=["mu", "tau"], extend=False)
print(summary_non_centered["rhat"])

az.plot_autocorr(centered_data, var_names=["mu", "tau"], combined=True)
az.plot_autocorr(non_centered_data, var_names=["mu", "tau"], combined=True)
plt.show()

# 3
divergences_centered = centered_data.sample_stats.diverging.sum()
print("\nNr divergente - Model centrat:", divergences_centered)


divergences_non_centered = non_centered_data.sample_stats.diverging.sum()
print("Nr divergente - Model necentrat:", divergences_non_centered)

az.plot_pair(centered_data, var_names=["mu", "tau"], divergences=True)
az.plot_pair(non_centered_data, var_names=["mu", "tau"], divergences=True)
plt.show()