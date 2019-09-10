#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 10:22:16 2019

@author: iagorosa
"""

import scipy.stats as scs
import numpy as np
import pandas as pd

# In[]
# Leitura
arq = open('./dados/carros.txt', 'r')
carros = arq.readlines()
carros = [int(c.strip('\n')) for c in carros]

# In[]
# Observacao
carros_obs = [ c if c < 3 else 3 for c in carros]
obs = [ carros_obs.count(i) for i in range(max(carros_obs)+1) ]
obs = np.array(obs)

# In[]

#y = [ carros.count(i) for i in range(max(carros)+1) ]

# Esperado

lbd = np.mean(carros) # aproximacao lambda
p = range(len(obs)) # quantidade de intervalos
esp = scs.poisson.pmf(p, lbd)[:-1] 
esp = np.append(esp, 1-sum(esp))
esp = esp*len(carros)

# In[]

alfa = 0.05
df = len(obs) - 2

X_2 = sum( (obs - esp)**2/esp )

q_alfa = scs.chi2.ppf(1-alfa, df)

p_val = 1-scs.chi2.cdf(X_2, df)



