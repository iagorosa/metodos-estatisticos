# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: iagorosa
"""

# In[]
import numpy as np 
import scipy.stats as scs
import pandas as pd
import pylab as pl
import seaborn as sns
from statsmodels.stats.diagnostic import lilliefors

from scipy.optimize import curve_fit
from scipy.special import factorial


pl.style.use('default')

#%%

arq = open('lowbw b.txt', 'r')

lowbw=[]
while True: 
    r = arq.readline().split() 
    if r != []:
        lowbw.append(r)
    else: 
        break
    
X = pd.DataFrame(lowbw[1:], columns=lowbw[0])

X = X.astype(int)

#%%

# 2.2.1. Descrição e modelagem dos dados
# 1 - a)
ptl = X.ptl
sns.countplot(ptl, color='b', ec='black', zorder=2)
# ptl.hist(histtype='bar', bins=range(0, 4), ec='black', zorder=2)
pl.xlabel('Histórico de partos prematuros',fontsize=12)
pl.ylabel('Quantidade', fontsize=12)
pl.show()
freq_ptl =  ptl.value_counts(sort=False)
freq_ptl = pd.DataFrame({"Freq": freq_ptl})
freq_ptl['Acum.'] = freq_ptl['Freq'].cumsum()
freq_ptl['Rel. Acum.'] = ((freq_ptl['Freq']/freq_ptl['Freq'].sum()).cumsum()*100).map('{:,.3f}'.format)
print()
print(freq_ptl)
print()

# 1 - b)
ftv = X.ftv
# ftv.hist(histtype='bar', bins=range(0, 7), ec='black', zorder=2)
sns.countplot(ftv, color='b', ec='black', zorder=2)
pl.xlabel('Numéro de visitas ao médico',fontsize=12)
pl.ylabel('Quantidade', fontsize=12)
pl.show()
freq_ftv =  ftv.value_counts(sort=False)
freq_ftv = pd.DataFrame({"Freq": freq_ftv})
freq_ftv['Acum.'] = freq_ftv['Freq'].cumsum()
freq_ftv['Rel. Acum.'] = ((freq_ftv['Freq']/freq_ftv['Freq'].sum()).cumsum()*100).map('{:,.3f}'.format)
print()
print(freq_ftv)
print()


#%%

# 2.2.1. Descrição e modelagem dos dados


# for i, d in enumerate(data):

def f_poisson(d, r, val_agrup, nome):
    # the bins should be of integer width, because poisson is an integer distribution
    entries, bin_edges, patches = pl.hist(d, bins=int(r[1]-r[0]), range=r, density=True, ec='black')

    # calculate binmiddles
    bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])

    # poisson function, parameter lamb is the fit parameter
    def poisson(k, lamb):
        return (lamb**k/factorial(k)) * np.exp(-lamb)

    # fit with curve_fit
    parameters, cov_matrix = curve_fit(poisson, bin_middles, entries) 

    # plot poisson-deviation with fitted parameter
    x_plot = np.linspace(0, r[1], 100)

    pl.plot(x_plot, poisson(x_plot, *parameters), 'r-', lw=2)
    pl.show()

    # CHI^2

    # vals_obs = [ c if c < 3 else 3 for c in d]
    vals_obs = [ c if c < val_agrup else val_agrup for c in d]
    obs = [ vals_obs.count(i) for i in range(max(vals_obs)+1) ]
    obs = np.array(obs)

    # Esperado 
    lbd = np.mean(d) # aproximacao lambda
    p = range(len(obs)) # quantidade de intervalos
    esp = scs.poisson.pmf(p, lbd)[:-1] 
    esp = np.append(esp, 1-sum(esp))
    esp = esp*len(d)

    alfa = 0.05
    df = len(obs) - 2

    X_2 = sum( (obs - esp)**2/esp )

    q_alfa = scs.chi2.ppf(1-alfa, df)

    p_val = 1-scs.chi2.cdf(X_2, df)

    print(nome+":", "X_2:", X_2, "q_alfa:", q_alfa, 'p_val:', p_val)


data = [ptl, ftv]
ranges = [[-0.5, 3.5], [-0.5, 6.5]]
nomes = ['ptl', 'ftv']


f_poisson(data[0], ranges[0], 3, nomes[0])
f_poisson(data[1], ranges[1], 4, nomes[1])


#%%

# 3 - a)

lwt = X.lwt
bwt = X.bwt

lwt.hist(histtype='bar', density=False, ec='black', zorder=2)
pl.xlabel("Peso da criança ao nascer (g)")
pl.ylabel("Frequência")
pl.grid(axis='x')
pl.show()

bwt.hist(histtype='bar', density=False, ec='black', zorder=2)
pl.xlabel("Peso da mãe na época da última menstruação (lb)")
pl.ylabel("Frequência")
pl.grid(axis='x')
pl.show()

desc = pd.DataFrame(lwt.describe())
desc.loc['skewness', 'lwt'] = scs.skew(lwt)
desc.loc['kurtosis', 'lwt'] = scs.kurtosis(lwt, fisher=False)

desc_aux = pd.DataFrame(bwt.describe())
desc_aux.loc['skewness', 'bwt'] = scs.skew(bwt)
desc_aux.loc['kurtosis', 'bwt'] = scs.kurtosis(bwt, fisher=False)

desc = pd.concat([desc, desc_aux], axis=1)

# %%

# 4)
lwt.hist(histtype='bar', density=True, ec='black', zorder=2)
pl.xlabel("Peso da criança ao nascer (g)")
pl.ylabel("Frequência")
pl.grid(axis='x')
# pl.xticks(range(10,101,10))

# estatistica
mu, std = scs.norm.fit(lwt)

# Plot the PDF.
xmin, xmax = pl.xlim()
x = np.linspace(xmin, xmax, 100)
p = scs.norm.pdf(x, mu, std)
pl.plot(x, p, 'r--', linewidth=2)

# Teste de hipotese de normalidade com 5% de significancia:
# H0: A amostra provem de uma população normal
# H1: A amostra nao provem de uma distribuicao normal

# Testes de shapiro e lillefors: 
s   = scs.shapiro(lwt)
lil = lilliefors(lwt)

pl.text(225, 0.018, 'Shapiro: '+str(round(s[1], 5) )+'\nLilliefors: '+str(round(lil[1], 5)), bbox=dict(facecolor='red', alpha=0.4), zorder=4 )

pl.show()



bwt.hist(histtype='bar', density=True, ec='black', zorder=2)
pl.xlabel("Peso da criança ao nascer (g)")
pl.ylabel("Frequência")
pl.grid(axis='x')
# pl.xticks(range(10,101,10))

# estatistica
mu, std = scs.norm.fit(bwt)

# Plot the PDF.
xmin, xmax = pl.xlim()
x = np.linspace(xmin, xmax, 100)
p = scs.norm.pdf(x, mu, std)
pl.plot(x, p, 'r--', linewidth=2)

# Teste de hipotese de normalidade com 5% de significancia:
# H0: A amostra provem de uma população normal
# H1: A amostra nao provem de uma distribuicao normal

# Testes de shapiro e lillefors: 
s   = scs.shapiro(bwt)
lil = lilliefors(bwt)

pl.text(xmin, 0.00052, 'Shapiro: '+str(round(s[1], 5) )+'\nLilliefors: '+str(round(lil[1], 5)), bbox=dict(facecolor='red', alpha=0.4), zorder=4 )

pl.show()



#%%

# 2.2.2. Transformações e recodificações dos dados

X['lwtkg'] = X['lwt']*0.453
X['race2'] = X['race'] == 1
X['ptl2'] = X['ptl'] >= 1
X['ftv2'] = X['ftv'] >= 1

X[['lwtkg', 'race2', 'ptl2', 'ftv2']].head()

#%%




#%%

a_ = pd.concat([X[['lwt', 'smoke']], pd.Series(['Fuma']*len(X))], axis=1)
b_ = pd.concat([X[['lwt', 'race2']], pd.Series(['Branca']*len(X))], axis=1)
c_ = pd.concat([X[['lwt', 'ptl2']], pd.Series(['Pre-Parto']*len(X))], axis=1)
d_ = pd.concat([X[['lwt', 'ht']], pd.Series(['Hipertensão']*len(X))], axis=1)
e_ = pd.concat([X[['lwt', 'ui']], pd.Series(['Irr-uterina']*len(X))], axis=1)
f_ = pd.concat([X[['lwt', 'ftv2']], pd.Series(['Consulta']*len(X))], axis=1)

frames = [a_, b_, c_, d_, e_, f_]

for f in frames:
    f.columns = ['lwt', 'sit', 'target']

D_ = pd.concate(frames)

#%%

pl.grid(ds='steps-mid', axis='y', zorder=5)
sns.violinplot(x='target', y='lwt', hue="sit",
data=D_, palette="muted", split=True, zorder=2)

#%%
