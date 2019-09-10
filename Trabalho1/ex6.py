#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 03:37:49 2019

@author: iagorosa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:43:36 2019

@author: iagorosa
"""

import pandas as pd
import numpy as np
import scipy as sc

# In[]


# Hipoteses
# H0 med_mac - med_fem = 0
# H1 med_mac - med_fem != 0

 
# In[]

# Amostra

#tab = [[3.64, 3.11, 3.8, 3.58, 4.55, 3.92], [1.91, 2.06, 1.78, 2.0, 1.3, 2.32]]
#df = pd.DataFrame(tab, index=['machos', 'femeas'], columns=np.arange(1,7))
#n = len(df.iloc[0, :])

am = np.array([37,36,37,30,37,39,40,47,36,42])
per = np.array([40,50,54,43,51,53,38,39,49,44,50,50,50,53,50,52,40,46,46,38])

n_am = len(am)
n_per = len(per)


# In[]

# Especificacao do teste

mu = 0          # media, de acordo com H0
alfa = 0.05/2     # nivel de significancia desejado
df = n_am + n_per - 2  # numero de graus de liberdade

# In[]

# Medias
med_am  = am.mean()
med_per = per.mean()

s_am  = sc.stats.tstd(am)
s_per = sc.stats.tstd(per)


# In[]

# teste de normalidade
# assumir hipotese que sao normais

print( sc.stats.shapiro(am)[1],  sc.stats.shapiro(per)[1] )
#print( (sc.stats.shapiro(am)[1] > alfa) and (sc.stats.shapiro(per)[1] > alfa) )

# In[]

# TESTE SE VARIANCIAS SAO DIFERENTES
# PELA APOSTILA, DIFERENTES SE UMA É 4X MAIOR QUE A OUTRA

print( (s_am**2 > 4*s_per**2) and (s_per**2 > 4*s_am**2) ) 
# Se True,  variancias diferentes
# Se False, variancias iguais

# Teste variancia
sc.stats.bartlett(am, per)

# In[]

# Estatisticas de teste
x_bar = med_am - med_per

s_comb = np.sqrt(  ( (n_am - 1)*s_am**2 + (n_per-1)*s_per**2 ) / df  )

t = (x_bar - mu) / (s_comb * np.sqrt(1/n_am + 1/n_per) )

print('t:', t) 

t_c = sc.stats.t.ppf(1-alfa, df)
p_val = 2*(1-sc.stats.t.cdf(abs(t), df))

print(t_c, p_val)
print(sc.stats.ttest_ind(am, per, equal_var=True))
