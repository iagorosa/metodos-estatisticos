#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:48:54 2019

@author: iagorosa
"""

import scipy.stats as scs
import numpy as np
import pandas as pd

# In[]

# Amostra

morte_feq = [109, 65, 22, 3, 1]


# In[]
# Observacao
#carros_obs = [ c if c < 3 else 3 for c in carros]
#obs = [ carros_obs.count(i) for i in range(max(carros_obs)+1) ]
#obs = np.array(obs)

morte_obs = morte_feq[:2] + [sum(morte_feq[2:])] 
morte_obs = np.array(morte_obs)

# In[]

#y = [ carros.count(i) for i in range(max(carros)+1) ]

# Esperado

lbd = np.mean(sum(i*morte_feq[i] for i in range(len(morte_feq)))/sum(morte_feq)) # aproximacao lambda
p = range(len(morte_obs)) # quantidade de intervalos
esp = scs.poisson.pmf(p, lbd)[:-1] 
esp = np.append(esp, 1-sum(esp))
esp = esp*sum(morte_feq)

# In[]

alfa = 0.05
df = len(morte_obs) - 2

X_2 = sum( (morte_obs - esp)**2/esp )

q_alfa = scs.chi2.ppf(1-alfa, df)




