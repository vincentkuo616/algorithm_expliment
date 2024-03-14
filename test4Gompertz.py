# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:12:04 2020

@author: vincentkuo
"""

from scipy.stats import gompertz
import matplotlib.pyplot as plt
import numpy as np
import pygrowthmodels
import math


fig, ax = plt.subplots(1, 1)

c = 0.947437130751
mean, var, skew, kurt = gompertz.stats(c, moments='mvsk')

x = np.linspace(gompertz.ppf(0.01, c),gompertz.ppf(0.99, c), 100)
ax.plot(x, gompertz.pdf(x, c),'r-', lw=5, alpha=0.6, label='gompertz pdf')

rv = gompertz(c)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

vals = gompertz.ppf([0.001, 0.5, 0.999], c)
np.allclose([0.001, 0.5, 0.999], gompertz.cdf(vals, c))

r = gompertz.rvs(c, size=1000)

ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()

data = np.array([122953,129218,137408,132525,123251,104393,99917,110559,141531,172787,199735,215132,233178,252342,274857,301251,323500,351600,364500,370000,387000])


print(pygrowthmodels.gompertz(0,99651.72,-math.log(1.23387),-math.log(1.236525)))
print(pygrowthmodels.gompertz(1,99651.72,-math.log(1.23387),-math.log(1.236525)))
print(pygrowthmodels.gompertz(2,99651.72,-math.log(1.23387),-math.log(1.236525)))
print(pygrowthmodels.gompertz(3,99651.72,-math.log(1.23387),-math.log(1.236525)))

