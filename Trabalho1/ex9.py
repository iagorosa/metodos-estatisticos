#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 16:31:05 2019

@author: iagorosa
"""

import scipy.stats as scs
import numpy as np
import pylab as pl
import pandas as pd
from statsmodels.stats.diagnostic import lilliefors

# In[]

arq = open('./dados/husbands.txt', 'r')
#husbands = pd.read_table('./dados/husbands.txt', sep=' ')


husbands=[]
while True: 
    r = arq.readline().split() 
    if r != []:
        husbands.append(r)
    else: 
        break
    
hus = pd.DataFrame(husbands[1:], columns=husbands[0])
hus = hus.astype(int)


# In[]
pl.figure()

# Idade marido
par=hus.ageh[hus.ageh>0]
lil = lilliefors(par)
par.hist(density=True, histtype='bar', ec='black', color='w')
#pl.text(19, 27, 'Shapiro: '+str(round(scs.shapiro(par)[1], 5) ) +'\nKS:'+str(round(scs.kstest(par, 'norm')[1], 5) ), bbox=dict(facecolor='red', alpha=0.1) )
pl.grid()
pl.text(17, 0.031, 'Shapiro: '+str(round(scs.shapiro(par)[1], 4) ) #+'\nKS: '+str(round(scs.kstest(par, 'norm')[1], 4))
+'\nLillie: '+str(round(lil[1], 4)), bbox=dict(facecolor='red', alpha=0.1) )
pl.title('Idade dos Maridos')
pl.xlabel('Idade')
pl.ylabel('Probabilidade')

#######
mu, std = scs.norm.fit(par)

# Plot the PDF.
xmin, xmax = pl.xlim()
x = np.linspace(xmin, xmax, 100)
p = scs.norm.pdf(x, mu, std)
pl.plot(x, p, 'r--', linewidth=2)

# In[]

pl.figure()
# Altura dos maridos
par=hus.heighth
lil = lilliefors(par)
par.hist(density=True, histtype='bar', ec='black', color='w')
#pl.text(1400, 40, 'Shapiro: '+str(round(scs.shapiro(par)[1], 5) ) +'\nKS:'+str(round(scs.kstest(par, 'norm')[1], 5) ), bbox=dict(facecolor='red', alpha=0.1) )
pl.grid()
pl.text(1530, 0.0062, 'Shapiro: '+str(round(scs.shapiro(par)[1], 4) ) #+'\nKS: '+str(round(scs.kstest(par, 'norm')[1], 4))
+'\nLillie: '+str(round(lil[1], 4)), bbox=dict(facecolor='red', alpha=0.1), )
pl.title('Altura dos Maridos')
pl.xlabel('Altura(mm)')
pl.ylabel('Probabilidade')

#######
mu, std = scs.norm.fit(par)

# Plot the PDF.
xmin, xmax = pl.xlim()
x = np.linspace(xmin, xmax, 100)
p = scs.norm.pdf(x, mu, std)
pl.plot(x, p, 'r--', linewidth=2)


# In[]
pl.figure()

# Idade esposas
par=hus.agew[hus.agew>0]
lil = lilliefors(par)
par.hist(density=True, histtype='bar', ec='black', color='w')
#pl.text(45, 65, 'Shapiro: '+str(round(scs.shapiro(par)[1], 5) ) +'\nKS:'+str(round(scs.kstest(par, 'norm')[1], 5) ), bbox=dict(facecolor='red', alpha=0.1) )
pl.grid()
pl.text(15, 0.034, 'Shapiro: '+str(round(scs.shapiro(par)[1], 4) ) #+'\nKS: '+str(round(scs.kstest(par, 'norm')[1], 4))
+'\nLillie: '+str(round(lil[1], 4)), bbox=dict(facecolor='red', alpha=0.1), zorder=4 )
pl.title('Idade das Esposas')
pl.xlabel('Idade')
pl.ylabel('Probabilidades')

#######
mu, std = scs.norm.fit(par)

# Plot the PDF.
xmin, xmax = pl.xlim()
x = np.linspace(xmin, xmax, 100)
p = scs.norm.pdf(x, mu, std)
pl.plot(x, p, 'r--', linewidth=2)

# In[]
pl.figure()

# Altura das mulheres
par=hus.height[hus.height>0]
lil = lilliefors(par)
par.hist(density=True, histtype='bar', ec='black', color='w')
#pl.text(1550, 48, 'Shapiro: '+str(round(scs.shapiro(par)[1], 5) ) +'\nKS:'+str(round(scs.kstest(par, 'norm')[1], 5) ), bbox=dict(facecolor='red', alpha=0.1) )
pl.grid()
pl.text(1385, 0.006, 'Shapiro: '+str(round(scs.shapiro(par)[1], 5) ) #+'\nKS: '+str(round(scs.kstest(par, 'norm')[1], 4))
+'\nLillie: '+str(round(lil[1], 4)), bbox=dict(facecolor='red', alpha=0.1), zorder=4 )
pl.title('Altura das Esposas')
pl.xlabel('Altura(mm)')
pl.ylabel('Probabilidades')

# find parameters
mu, std = scs.norm.fit(par)  # Par eh a minha distibuicao

# Plot the PDF.
xmin, xmax = pl.xlim()
x = np.linspace(xmin, xmax, 100)
p = scs.norm.pdf(x, mu, std)
pl.plot(x, p, 'r--', linewidth=2)
# In[]
pl.figure()

#Idade dos maridos antes de casar

par=hus.agehm[hus.agehm>0]
lil = lilliefors(par)
par.hist(density=True, histtype='bar', ec='black', color='w')
#pl.text(17, 26, 'Shapiro: '+str(round(scs.shapiro(par)[1], 5) ) +'\nKS:'+str(round(scs.kstest(par, 'norm')[1], 5) ), bbox=dict(facecolor='red', alpha=0.1) )
pl.grid()
pl.text(46, 0.096, 'Shapiro: '+str(round(scs.shapiro(par)[1], 5) ) #+'\nKS: '+str(round(scs.kstest(par, 'norm')[1], 4))
+'\nLillie: '+str(round(lil[1], 4)), bbox=dict(facecolor='red', alpha=0.1), zorder=4 )
pl.title('Idade dos Maridos Antes de Casar')
pl.xlabel('Idade')
pl.ylabel('Probabilidades')

## Fit a normal distribution to the data:
mu, std = scs.norm.fit(par)

# Plot the PDF.
xmin, xmax = pl.xlim()
x = np.linspace(xmin, xmax, 100)
p = scs.norm.pdf(x, mu, std)
pl.plot(x, p, 'r--', linewidth=2)

# In[]



print('shapiro:', scs.shapiro(hus.ageh)[1], '\nKS:', scs.kstest(hus.ageh, 'norm')[1] )
#pl.xticks(range(0, max(hus.ageh), 10))


from statsmodels.stats.diagnostic import lilliefors

# Teste de hipotese com 5% de significancia:
# H0: A amostra provem de uma população normal
# H1: A amostra nao provem de uma distribuicao normal

# Testes de shapiro e lillefors: 
scs.shapiro(par)
lilliefors(par)

# Se o p-valor encontrado for menor que 5%, rejeita a hipotese nula
# Se o p-valor encontrado for maior que 5%, não há evidências para rejeitar a hipotese nula
# e pode-se acreditar que a populacao proveniente da amostra testada eh ajustada por uma distribuicao normal




