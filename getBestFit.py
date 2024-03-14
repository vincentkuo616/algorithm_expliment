
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 14:54:08 2020

@author: hungyutsai
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import time 
from time import localtime, strftime
import warnings
import scipy.stats as st
import matplotlib
import math
from scipy.optimize import leastsq
from scipy.stats import truncnorm


#matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

warnings.filterwarnings('ignore')
prjRootFolder=r"C:\\Users\\vincentkuo\\Documents\\"


print("\n ===== Main Start ===== ")
MainStartTime = time.time()
sessionName='Loading'
print("[{} {}] {} ".format(sessionName,"Start",strftime("%Y-%m-%d %H:%M:%S", localtime())))

readfile='經典手刻90_out'

data= pd.read_excel(f"{prjRootFolder}/{readfile}.xlsx",encoding='utf-8')
dfArray=data.values
df=data['Hello']


# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    # 特別慢的先不跑 st.levy_stable,
    DISTRIBUTIONS = [        
        
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,

        
        #st.genpareto,
        #st.genexpon
        #,st.gilbrat,st.halfcauchy,st.halfgennorm,st.kappa3,st.loguniform,st.lomax,st.pareto,st.truncexpon
    ]
    
    
    
    
    
    '''
    
    st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    '''
    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf
    mu = np.inf
    var= np.inf
    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:
        # Try to fit the distribution
        try:
            
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                # fit dist to data
                data = np.array(data)
                #fit會回傳多個參數，那可以確定的是，最後一個是變異數，倒數第二個是平均
                params = distribution.fit(data)
                
                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                mae = np.sum(abs(y - pdf))
                print(f'{distribution.name}, SSE = {sse}')
                print(f'{distribution.name}, mae = {mae}')
                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass
                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    mu = loc
                    var = scale
                    best_params = params
                    best_sse = sse
            
        except Exception  :
            pass

    return (best_distribution.name, best_params,mu,var)

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]   # 平均
    scale = params[-1] #  變異數

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

# Load data from statsmodels datasets
#data = pd.Series(sm.datasets.elnino.load_pandas().data.set_index('YEAR').values.ravel())

data = df
#st.halfgennorm(beta=0.878020, loc=1.0, scale=0.919862)

# Plot for comparison
plt.figure(figsize=(9,6))
ax = data.plot(kind='hist', bins=50, alpha=0.5)
# Save plot limits
dataYLim = ax.get_ylim()

# Find best fit distribution
best_fit_name, best_fit_params,mean,var = best_fit_distribution(data, 200, ax)
print('mean=',mean)
print('std=',math.sqrt(var))
best_dist = getattr(st, best_fit_name)

# Update plots

ax.set_ylim(dataYLim)
ax.set_title(u'All Fitted Distributions')
ax.set_xlabel(u'X')
ax.set_ylabel('Frequency')

# Make PDF with best params 
pdf = make_pdf(best_dist, best_fit_params)

# Display
plt.figure(figsize=(9,6))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data.plot(kind='hist', bins=300,  alpha=0.5, label='Data', legend=True, ax=ax)

param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
dist_str = '{}({})'.format(best_fit_name, param_str)


ax.set_title(u' with best fit distribution \n' + dist_str)
ax.set_xlabel(u'X')
ax.set_ylabel('Frequency')





'''
from scipy.stats import halfgennorm
outputArray=np.array(outputList)
beta,loc,scale=halfgennorm.fit(outputArray)
print(f"beta={beta}, loc={loc}, scale={scale}")


fig, ax = plt.subplots(1, 1)

x = np.linspace(halfgennorm.ppf(0.01, beta),
                halfgennorm.ppf(0.99, beta), 100)
ax.plot(x, halfgennorm.pdf(x, beta),
       'r-', lw=5, alpha=0.6, label='halfgennorm pdf')
rv = halfgennorm(beta=beta, loc=loc,scale=scale)


ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
plt.show()
r = halfgennorm.rvs(beta, size=1000)

vals = halfgennorm.ppf([0.001, 0.5, 0.999], beta)
print(np.allclose([0.001, 0.5, 0.999], halfgennorm.cdf(vals, beta)))

ax.hist(outputArray, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()
'''