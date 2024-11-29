import scipy.stats as stats
import numpy as np

# Your actual data
returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])

# Test if mean is different from 0
# H0: mean = 0
# H1: mean ≠ 0
t_stat, p_value = stats.ttest_1samp(returns, popmean=0)
