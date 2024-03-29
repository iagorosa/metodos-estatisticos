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

import statsmodels.api as sm
from statsmodels.formula.api import ols



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
pl.grid(axis='y')
# ptl.hist(histtype='bar', bins=range(0, 4), ec='black', zorder=2)
pl.xlabel('ptl',fontsize=12)
pl.ylabel('Quantidade', fontsize=12)
pl.title("Quantidade de casos do histórico de partos prematuros")
pl.savefig("imgs/lowbw/barplot-ptl.pdf")
pl.show()
freq_ptl =  ptl.value_counts(sort=False)
freq_ptl = pd.DataFrame({"Freq": freq_ptl})
freq_ptl['Acum.'] = freq_ptl['Freq'].cumsum()
freq_ptl['Rel. Acum.'] = ((freq_ptl['Freq']/freq_ptl['Freq'].sum()).cumsum()*100).map('{:,.3f}'.format)
print()
print(freq_ptl)
print()2

# 1 - b)
ftv = X.ftv
# ftv.hist(histtype='bar', bins=range(0, 7), ec='black', zorder=2)
sns.countplot(ftv, color='b', ec='black', zorder=2)
pl.grid(axis='y')
pl.xlabel('ftv',fontsize=12)
pl.ylabel('Quantidade', fontsize=12)
pl.title("Quantidade de número de visitas ao médico")
pl.savefig("imgs/lowbw/barplot-ftv.pdf")
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
    entries, bin_edges, patches = pl.hist(d, bins=int(r[1]-r[0]), range=r, density=True, ec='black', zorder=2)

    # calculate binmiddles
    bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])

    # poisson function, parameter lamb is the fit parameter
    def poisson(k, lamb):
        return (lamb**k/factorial(k)) * np.exp(-lamb)

    # fit with curve_fit
    parameters, cov_matrix = curve_fit(poisson, bin_middles, entries) 

    # plot poisson-deviation with fitted parameter
    x_plot = np.linspace(0, r[1], 100)

    pl.plot(x_plot, poisson(x_plot, *parameters), 'r-', lw=2, zorder=2)
    pl.grid(axis='y', zorder=0)
    pl.xlabel(nome)
    pl.ylabel("Probabilidade")
    pl.title("Densidade de probabilidade de " + nome + " com modelo de Poisson")
    pl.savefig("imgs/lowbw/dens_poisson_"+nome+".pdf")
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

    print(lbd, df)
    print(obs)
    print(esp)
    print()


    X_2 = sum( (obs - esp)**2/esp )

    q_alfa = scs.chi2.ppf(1-alfa, df)

    p_val = 1-scs.chi2.cdf(X_2, df)

    print(nome+":", "X_2:", X_2, "q_alfa:", q_alfa, 'p_val:', p_val)


data = [ptl, ftv]
ranges = [[-0.5, 3.5], [-0.5, 6.5]]
nomes = ['ptl', 'ftv']


f_poisson(data[0], ranges[0], 2, nomes[0])
f_poisson(data[1], ranges[1], 4, nomes[1])


#%%

# 3 - a)

lwt = X.lwt
bwt = X.bwt

lwt.hist(histtype='bar', density=False, ec='black', zorder=2)
pl.xticks(range(80, 251, 17))
pl.xlabel("Peso da mãe na época da última menstruação (lb)")
pl.ylabel("Frequência")
pl.title("Histograma do peso da mãe na última menstruação")
pl.grid(axis='x')
pl.savefig("imgs/lowbw/hist_lwt.pdf")
pl.show()

bwt.hist(histtype='bar', density=False, ec='black', zorder=2)
pl.xticks(range(709, 5010, 428))
pl.xlabel("Peso da criança ao nascer (g)")
pl.ylabel("Frequência")
pl.title("Histograma do peso da criança ao nascer")
pl.grid(axis='x')
pl.savefig("imgs/lowbw/hist_bwt.pdf")
pl.show()

desc = pd.DataFrame(lwt.describe())
desc.loc['skewness', 'lwt'] = scs.skew(lwt)
desc.loc['kurtosis', 'lwt'] = scs.kurtosis(lwt, fisher=False)

desc_aux = pd.DataFrame(bwt.describe())
desc_aux.loc['skewness', 'bwt'] = scs.skew(bwt)
desc_aux.loc['kurtosis', 'bwt'] = scs.kurtosis(bwt, fisher=False)

desc = pd.concat([desc, desc_aux], axis=1)
desc = desc.applymap('{:,.3f}'.format)

# %%

# 4)
lwt.hist(histtype='bar', density=True, ec='black', zorder=2)
pl.xticks(range(80, 251, 17))
pl.xlabel("Peso da criança ao nascer (g)")
pl.ylabel("Frequência")
pl.title("Ajuste de modelo normal à variável lwt")
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

pl.savefig("imgs/lowbw/ajuste_normal_lwt.pdf")
pl.show()



bwt.hist(histtype='bar', density=True, ec='black', zorder=2)
pl.xticks(range(709, 5010, 428))
pl.xlabel("Peso da mãe na época da última menstruação (lb)")
pl.ylabel("Frequência")
pl.title("Ajuste de modelo normal a variável bwt")
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

pl.savefig("imgs/lowbw/ajuste_normal_bwt.pdf")
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

a_ = pd.concat([X[['bwt', 'smoke']], pd.Series(['Fuma']*len(X))], axis=1)
b_ = pd.concat([X[['bwt', 'race2']], pd.Series(['Branca']*len(X))], axis=1)
c_ = pd.concat([X[['bwt', 'ptl2']], pd.Series(['Pre-Parto']*len(X))], axis=1)
d_ = pd.concat([X[['bwt', 'ht']], pd.Series(['Hipertensão']*len(X))], axis=1)
e_ = pd.concat([X[['bwt', 'ui']], pd.Series(['Irr-uterina']*len(X))], axis=1)
f_ = pd.concat([X[['bwt', 'ftv2']], pd.Series(['Consulta']*len(X))], axis=1)

frames = [a_, b_, c_, d_, e_, f_]

for f in frames:
    f.columns = ['lwt', 'sit', 'target']

D_ = pd.concat(frames)

#%%

pl.grid(ds='steps-mid', axis='y', zorder=5)
sns.violinplot(x='target', y='lwt', hue="sit",
data=D_, palette="muted", split=True, zorder=2)

#%%

# 2.2.3. Análise usando bwt (variável quantitativa)

vars = ['smoke', 'race2', 'ptl2', 'ht', 'ui', 'ftv2']

f_trues = []
f_falses = []
for v in vars: ax = sns.boxplot(x="day", y="total_bill", hue="smoker",
...                  data=tips, palette="Set3")
    f_trues.append(X[X[v] == True]['bwt'])
    f_falses.append(X[X[v] == False]['bwt'])
    pl.boxplot([f_falses[-1], f_trues[-1]])
    pl.xticks([1,2], [0, 1])
    pl.yticks(range(0, 5001, 500))
    # pl.grid(axis='y')
    pl.show()
    pl.close()

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
pl.grid(axis='y')
# ptl.hist(histtype='bar', bins=range(0, 4), ec='black', zorder=2)
pl.xlabel('ptl',fontsize=12)
pl.ylabel('Quantidade', fontsize=12)
pl.title("Quantidade de casos do histórico de partos prematuros")
pl.savefig("imgs/lowbw/barplot-ptl.pdf")
pl.show()
freq_ptl =  ptl.value_counts(sort=False)
freq_ptl = pd.DataFrame({"Freq": freq_ptl})
freq_ptl['Acum.'] = freq_ptl['Freq'].cumsum()
freq_ptl['Rel. Acum.'] = ((freq_ptl['Freq']/freq_ptl['Freq'].sum()).cumsum()*100).map('{:,.3f}'.format)
print()
print(freq_ptl)
print()2

# 1 - b)
ftv = X.ftv
# ftv.hist(histtype='bar', bins=range(0, 7), ec='black', zorder=2)
sns.countplot(ftv, color='b', ec='black', zorder=2)
pl.grid(axis='y')
pl.xlabel('ftv',fontsize=12)
pl.ylabel('Quantidade', fontsize=12)
pl.title("Quantidade de número de visitas ao médico")
pl.savefig("imgs/lowbw/barplot-ftv.pdf")
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
    entries, bin_edges, patches = pl.hist(d, bins=int(r[1]-r[0]), range=r, density=True, ec='black', zorder=2)

    # calculate binmiddles
    bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])

    # poisson function, parameter lamb is the fit parameter
    def poisson(k, lamb):
        return (lamb**k/factorial(k)) * np.exp(-lamb)

    # fit with curve_fit
    parameters, cov_matrix = curve_fit(poisson, bin_middles, entries) 

    # plot poisson-deviation with fitted parameter
    x_plot = np.linspace(0, r[1], 100)

    pl.plot(x_plot, poisson(x_plot, *parameters), 'r-', lw=2, zorder=2)
    pl.grid(axis='y', zorder=0)
    pl.xlabel(nome)
    pl.ylabel("Probabilidade")
    pl.title("Densidade de probabilidade de " + nome + " com modelo de Poisson")
    pl.savefig("imgs/lowbw/dens_poisson_"+nome+".pdf")
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

    print(lbd, df)
    print(obs)
    print(esp)
    print()


    X_2 = sum( (obs - esp)**2/esp )

    q_alfa = scs.chi2.ppf(1-alfa, df)

    p_val = 1-scs.chi2.cdf(X_2, df)

    print(nome+":", "X_2:", X_2, "q_alfa:", q_alfa, 'p_val:', p_val)


data = [ptl, ftv]
ranges = [[-0.5, 3.5], [-0.5, 6.5]]
nomes = ['ptl', 'ftv']


f_poisson(data[0], ranges[0], 2, nomes[0])
f_poisson(data[1], ranges[1], 4, nomes[1])


#%%

# 3 - a)

lwt = X.lwt
bwt = X.bwt

lwt.hist(histtype='bar', density=False, ec='black', zorder=2)
pl.xticks(range(80, 251, 17))
pl.xlabel("Peso da mãe na época da última menstruação (lb)")
pl.ylabel("Frequência")
pl.title("Histograma do peso da mãe na última menstruação")
pl.grid(axis='x')
pl.savefig("imgs/lowbw/hist_lwt.pdf")
pl.show()

bwt.hist(histtype='bar', density=False, ec='black', zorder=2)
pl.xticks(range(709, 5010, 428))
pl.xlabel("Peso da criança ao nascer (g)")
pl.ylabel("Frequência")
pl.title("Histograma do peso da criança ao nascer")
pl.grid(axis='x')
pl.savefig("imgs/lowbw/hist_bwt.pdf")
pl.show()

desc = pd.DataFrame(lwt.describe())
desc.loc['skewness', 'lwt'] = scs.skew(lwt)
desc.loc['kurtosis', 'lwt'] = scs.kurtosis(lwt, fisher=False)

desc_aux = pd.DataFrame(bwt.describe())
desc_aux.loc['skewness', 'bwt'] = scs.skew(bwt)
desc_aux.loc['kurtosis', 'bwt'] = scs.kurtosis(bwt, fisher=False)

desc = pd.concat([desc, desc_aux], axis=1)
desc = desc.applymap('{:,.3f}'.format)

# %%

# 4)
lwt.hist(histtype='bar', density=True, ec='black', zorder=2)
pl.xticks(range(80, 251, 17))
pl.xlabel("Peso da criança ao nascer (g)")
pl.ylabel("Frequência")
pl.title("Ajuste de modelo normal à variável lwt")
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

pl.savefig("imgs/lowbw/ajuste_normal_lwt.pdf")
pl.show()



bwt.hist(histtype='bar', density=True, ec='black', zorder=2)
pl.xticks(range(709, 5010, 428))
pl.xlabel("Peso da mãe na época da última menstruação (lb)")
pl.ylabel("Frequência")
pl.title("Ajuste de modelo normal a variável bwt")
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

pl.savefig("imgs/lowbw/ajuste_normal_bwt.pdf")
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

a_ = pd.concat([X[['bwt', 'smoke']], pd.Series(['Fuma']*len(X))], axis=1)
b_ = pd.concat([X[['bwt', 'race2']], pd.Series(['Branca']*len(X))], axis=1)
c_ = pd.concat([X[['bwt', 'ptl2']], pd.Series(['Pre-Parto']*len(X))], axis=1)
d_ = pd.concat([X[['bwt', 'ht']], pd.Series(['Hipertensão']*len(X))], axis=1)
e_ = pd.concat([X[['bwt', 'ui']], pd.Series(['Irr-uterina']*len(X))], axis=1)
f_ = pd.concat([X[['bwt', 'ftv2']], pd.Series(['Consulta']*len(X))], axis=1)

frames = [a_, b_, c_, d_, e_, f_]

for f in frames:
    f.columns = ['lwt', 'sit', 'target']

D_ = pd.concat(frames)

#%%

pl.grid(ds='steps-mid', axis='y', zorder=5)
sns.violinplot(x='target', y='lwt', hue="sit",
data=D_, palette="muted", split=True, zorder=2)

#%%

# 2.2.3. Análise usando bwt (variável quantitativa)

vars = ['smoke', 'race2', 'ptl2', 'ht', 'ui', 'ftv2']

f_trues = []
f_falses = []
for v in vars: ax = sns.boxplot(x="day", y="total_bill", hue="smoker",
...                  data=tips, palette="Set3")
    f_trues.append(X[X[v] == True]['bwt'])
    f_falses.append(X[X[v] == False]['bwt'])
    pl.boxplot([f_falses[-1], f_trues[-1]])
    pl.xticks([1,2], [0, 1])
    pl.yticks(range(0, 5001, 500))
    # pl.grid(axis='y')
    pl.show()
    pl.close()


# pl.boxplot()

#%%

sns.boxplot(x='target', y='lwt', hue="sit", data=D_, palette="Set3", zorder=1)
pl.yticks(range(0, 5001, 500))
pl.legend(title='label',loc='upper center')
# pl.grid(axis='y)
pl.xlabel("Variável")
pl.ylabel("Peso da criança ao nascer (g)")
pl.title("Boxplots comparativos das variáveis binárias")
pl.savefig("imgs/lowbw/boxplots_comparativos_var_bin.pdf")
pl.show()
pl.close()


#%%

# pl.boxplot()

#%%

sns.boxplot(x='target', y='lwt', hue="sit", data=D_, palette="Set3", zorder=1)
pl.yticks(range(0, 5001, 500))
pl.legend(title='label',loc='upper center')
# pl.grid(axis='y)
pl.xlabel("Variável")
pl.ylabel("Peso da criança ao nascer (g)")
pl.title("Boxplots comparativos das variáveis binárias")
pl.savefig("imgs/lowbw/boxplots_comparativos_var_bin.pdf")
pl.show()
pl.close()

#%%




#%%


mod = ols('bwt ~ C(race)', data=X).fit()
anova_table_bwt = sm.stats.anova_lm(mod, typ=2)
anova_table_bwt = (anova_table_bwt.replace({np.nan: 0})).applymap('{:,.3f}'.format)
print()
display(anova_table_bwt)
print()

#%%

# sns.boxplot(x='race', y='bwt', data=D_, palette="Set3", zorder=1)
pl.boxplot([X[X.race == 1]['bwt'], X[X.race == 2]['bwt'], X[X.race == 3]['bwt']])


#%%

def teste_media(med_x, med_y, s_x, s_y, n_x, n_y, mu, alfa):
    df = n_x + n_y - 2
    # Estatisticas de teste
    x_bar = abs(med_x - med_y)

    s_comb = np.sqrt(  ( (n_x - 1)*s_x**2 + (n_y-1)*s_y**2 ) / df  )
    # print(x_bar, s_x, s_y, s_comb)

    z = (x_bar - mu) / (s_comb * np.sqrt(1/n_x + 1/n_y) )

    # print('t:', t) 

    z_c = scs.norm.ppf(1-alfa)
    p_val = 2*(1-scs.norm.cdf(z))

    print(round(z, 3), round(z_c, 3), round(p_val, 3))
    # print(sc.stats.ttest_ind(machos, femeas, equal_var=True))


# z_idoso = (mu_obs - mu)/(std / len(c[c['jovem'] == False]['N'])**0.5)

# z_c_idoso = scs.norm.ppf(1-alpha)
# p_val_idoso =(1-scs.norm.cdf(z_idoso))

vars = ['smoke', 'race2', 'ptl2', 'ht', 'ui', 'ftv2']

for v in vars:
    fm = X[X[v] == True]['bwt']
    nfm = X[X[v] == False]['bwt']
    mu = 0
    alfa = 0.05
    print(v+":", end=' ')
    teste_media(fm.mean(), nfm.mean(), fm.std(), nfm.std(), len(fm), len(nfm), mu, alfa)

#%%

mod = ols('bwt ~ lwt', data=X).fit()
anova_table_lwt = sm.stats.anova_lm(mod, typ=2)
anova_table_lwt = (anova_table_lwt.replace({np.nan: 0})).applymap('{:,.3f}'.format)
print()
display(anova_table_lwt)
print()

#%%

mod = ols('bwt ~ age', data=X).fit()
anova_table_age = sm.stats.anova_lm(mod, typ=2)
anova_table_age = (anova_table_age.replace({np.nan: 0})).applymap('{:,.3f}'.format)
print()
display(anova_table_age)
print()

#%%

pl.scatter(X['bwt'], X['lwt'], ec='black')
pl.xticks(range(1000, 5001, 500))
pl.ylabel("lwt")
pl.xlabel("bwt")
# pl.title("Gráfico de dispersão bwt vs lwt")
pl.title("Gráfico do peso ao nascer por peso da mãe")

cor = X[['bwt', 'lwt']].corr().iloc[1, 0]
pl.text(800, 240, "Correlação: " + str(round(cor, 3)), bbox=dict(facecolor='red', alpha=0.4))

# a1, b1, *resto1 = scs.linregress(X['bwt'], X['lwt'])
# x_1 = np.linspace(X['bwt'].min(), X['bwt'].max(), 100)
# pl.plot(x_1, a1*x_1+b1, 'red')
# pl.text(800,230, r'$Y = \beta X + \alpha$' +'\nC = '+str(round(a1, 3))+'I+'+str(round(b1, 3)), bbox=dict(facecolor='red', alpha=0.4))
pl.savefig("imgs/lowbw/scatter_bwt_lwt.pdf")
pl.show()

    

pl.scatter(X['bwt'], X['age'], ec='black')
pl.xticks(range(1000, 5001, 500))
pl.ylabel("age")
pl.xlabel("bwt")
pl.title("Gráfico da idade por peso da mãe")

cor = X[['bwt', 'age']].corr().iloc[1, 0]
pl.text(800, 43, "Correlação: " + str(round(cor, 3)), bbox=dict(facecolor='red', alpha=0.4))

# a1, b1, *resto1 = scs.linregress(X['bwt'], X['age'])
# x_1 = np.linspace(X['bwt'].min(), X['bwt'].max(), 100)
# pl.plot(x_1, a1*x_1+b1, 'red')
# pl.text(800,41, r'$Y = \beta X + \alpha$' +'\nC = '+str(round(a1, 3))+'I+'+str(round(b1, 3)), bbox=dict(facecolor='red', alpha=0.4))
pl.savefig("imgs/lowbw/scatter_bwt_age.pdf")
pl.show()

corr = X[['bwt', 'age']].corr()
print(corr)


#%%
