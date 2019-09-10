#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 18:16:30 2019

@author: iagorosa
"""

import scipy.stats as scs
import numpy as np
import pylab as pl
import pandas as pd
from statsmodels.stats.diagnostic import lilliefors

# In[]

arq = open('./dados/Michelson.txt', 'r')
mich = arq.readlines()
mich = [int(c.strip('\n')) for c in mich]


# In[]

pl.figure()
arq = open('./dados/Michelson.txt', 'r')
mich = arq.readlines()
mich = [int(c.strip('\n')) for c in mich]

n, bins, patches = pl.hist(mich, density=True, histtype='bar', ec='black', color='w')
pl.text(585, 0.0055, 'Shapiro: '+str(round(scs.shapiro(mich)[1], 4) ) #+'\nKS:'+str(round(scs.kstest(mich, 'norm')[1], 4))
+'\nLillie: '+str(round(lilliefors(mich)[1], 4) ), bbox=dict(facecolor='red', alpha=0.1) )
pl.grid(False)

pl.title('Densidade de Probabilidade -- Velocidades Medidas')
pl.xlabel('Velocidades Medidas')
pl.ylabel('Probabilidades')


# Fit a normal distribution to the data:
mu, std = scs.norm.fit(mich)

# Plot the PDF.
xmin, xmax = pl.xlim()
x = np.linspace(xmin, xmax, 100)
p = scs.norm.pdf(x, mu, std)
pl.plot(x, p, 'r--', linewidth=2)
#title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
#pl.title('tt')

