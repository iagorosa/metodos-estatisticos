#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 17:45:21 2019

@author: iagorosa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:49:45 2019

@author: iagorosa
"""

import scipy.stats as scs
import numpy as np

# In[]

#x = np.array([111, 119, 121, 113, 116, 126, 128, 123, 122, 121])
#y = np.array([109, 113, 120, 117, 108, 120, 122, 124, 115, 112])

#n_sim = len(sim)
#n_nao = len(nao)

d = np.array([-10, -25, -20, -5, -10, -20])
n = len(d)

# In[]



# Hipoteses
# H0 med_mac - med_fem = 0
# H1 med_mac - med_fem != 0

# In[]

# Especificacao do teste

mu = 0          # media, de acordo com H0
alfa = 0.05/2     # nivel de significancia desejado
#df = n_sim + n_nao - 2  # numero de graus de liberdade
df = len(d) - 1

# In[]

#med_sim = sim.mean()
#med_nao = nao.mean()

#s_sim = scs.tstd(sim)
#s_nao = scs.tstd(nao)

s_d = scs.tstd(d)

# In[]

# Teste de normalidade

#print(scs.shapiro(sim))
#print(scs.shapiro(nao))
#print(scs.kstest(sim, 'norm'))


# In[]

# Teste de variancia iguais

#print(s_sim**2/s_nao**2)
#print(s_nao**2/s_sim**2)

# In[]

# Estatisticas de teste
#x_bar = med_sim - med_nao
x_bar = d.mean()

#s_comb = np.sqrt(  ( (n_sim - 1)*s_sim**2 + (n_nao-1)*s_nao**2 ) / df  )

t = (x_bar - mu) / (s_d / np.sqrt(len(d)) )

print('t:', t) 

t_c = scs.t.ppf(1-alfa, df)
p_val = 2*(1-scs.t.cdf(t, df))

print(t_c, p_val)
#print(scs.ttest_ind(sim, nao, equal_var=True))








