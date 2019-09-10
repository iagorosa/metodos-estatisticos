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

machos = np.array([3.64, 3.11, 3.80, 3.58, 4.55, 3.92])
femeas = np.array([1.91, 2.06, 1.78, 2.00, 1.30, 2.32])

n_m = len(machos)
n_f = len(femeas)

#diferenca = machos - femeas
#n = len(diferenca) 

# In[]

# Especificacao do teste

mu = 0          # media, de acordo com H0
alfa = 0.05/2     # nivel de significancia desejado
df = n_m + n_f - 2  # numero de graus de liberdade
#tc_m = sc.stats.t.ppf(1-alfa, df)

# In[]

# Medias
#med_mac = df.loc['machos'].mean()
#med_fem = df.loc['femeas'].mean()

med_mac = machos.mean()
med_fem = femeas.mean()
#dp_mac = machos.std()
#dp_fem = femeas.std()
s_m = sc.stats.tstd(machos)
s_f = sc.stats.tstd(femeas)

# In[]

# teste de normalidade
# assumir hipotese que sao normais

#print(sc.stats.kstest(machos, 'norm')[1] < alfa) and (sc.stats.kstest(femeas, 'norm')[1] < alfa)    # Imprime verdadeiro ou faso para a normalidade
print( (sc.stats.shapiro(machos)[1] > alfa) and (sc.stats.shapiro(femeas)[1] > alfa) )

# In[]

# TESTE SE VARIANCIAS SAO DIFERENTES
# PELA APOSTILA, DIFERENTES SE UMA É 4X MAIOR QUE A OUTRA

print( (s_m**2 > 4*s_f**2) and (s_f**2 > 4*s_m**2) ) 
# Se True,  variancias diferentes
# Se False, variancias iguais


# In[]

# Diferencas

#x = diferenca
#x_dif = diferenca.mean()
#s_dif = diferenca.std()

# In[]

#MEDINA

s_comb = np.sqrt(  ( (n_m - 1)*s_m**2 + (n_f-1)*s_f**2 ) / df  )

t = sc.stats.t.ppf(1-alfa, df) # p-valor da dist. t-student
t_c =  t*s_comb + mu

######


# In[]

# Estatisticas de teste
x_bar = med_mac - med_fem

s_comb = np.sqrt(  ( (n_m - 1)*s_m**2 + (n_f-1)*s_f**2 ) / df  )

t = (x_bar - mu) / (s_comb * np.sqrt(1/n_m + 1/n_f) )

print('t:', t) 

t_c = sc.stats.t.ppf(1-alfa, df)
p_val = 2*(1-sc.stats.t.cdf(t, df))

print(t_c, p_val)
print(sc.stats.ttest_ind(machos, femeas, equal_var=True))

# In[]

'''
import scipy as sc
D_bar = X_bar - Y_bar    # Calcula diferença das médias entre X e Y
df = n_1 + n_2 - 2  # Graus de liberdade

S_comb = (  ( (n_1 - 1)*S_x**2 + (n_2-1)*S_y**2 ) / df  )**0.5  # Variância combinada

t = (D_bar - ED_bar) / (S_comb * np.sqrt(1/n_1 + 1/n_2) )  # Transforma os valores para a t-student

p_val = 2*(1-sc.stats.t.cdf(t, df))   # Verifica a integral t-student até o ponto t encontrado
'''