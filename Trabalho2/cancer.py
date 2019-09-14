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
import statsmodels.api as sm
from statsmodels.formula.api import ols

pl.style.use('default')
# pl.style.use('./PlotStyle.mplstyle')

# In[]

arq = open('cancer.txt', 'r')

cancer=[]
while True: 
    r = arq.readline().split() 
    if r != []:
        cancer.append(r)
    else: 
        break
    
c = pd.DataFrame(cancer[1:], columns=cancer[0])


c[['Ident', 'Diagn', 'Idade', 'ALB', 'N', 'GL']] = c[['Ident', 'Diagn', 'Idade', 'ALB', 'N', 'GL']].astype(int)
c[['AKP', 'P', 'LDH']] = c[['AKP', 'P', 'LDH']].astype(float)

# In[]

'''
	ident: Identificação do paciente.
	Diagn: Diagnóstico: 
		1 = Falso-negativo: diagnosticados como não tendo a 
			doença quando na verdade a tinham. 
		2 = Negativo: diagnosticados como não tendo a doença 
			quando de fato não não a tinham. 
		3 = Positivo: diagnosticados corretamente como tendo a doença.
		4=Falso-positivo: diagnosticados como tendo a doença quando na 
			verdade não tinham.
	IDade: Idade.
	AKP: Espectro químico da análise do sangue-alkaliine phosphatose (AKP). 
	P: Concentração de fosfato no sangue (P).
	LDH: Enzima, lactate dehydrogenase (LDH).
	ALB: Albumina (ALB).
	N: Nitrogênio na uréia (N).
	GL: Glicose (GL).
'''

#%%

# EXERCICIO 24b - pag. 45

# Histograma de idade dos falso-positivos
fp = c[c['Diagn'] == 4]['Idade']
bins_range = range(0, fp.max()+10,10)
out_fp = pd.cut(fp, bins=bins_range, include_lowest=True, right=False)

fp.hist(histtype='bar', bins=range(0,120,10), ec='black', zorder=2)
pl.grid(axis='x')
pl.title('Idade dos falso-positivos', fontsize=14)
pl.xlabel('Idade', fontsize=12)
pl.ylabel('Frequência', fontsize=12)
pl.xticks(range(10,120,10))
pl.savefig("./imgs/cancer/hist_falso-positivos.pdf", dpi=300)
pl.show()


# Histograma de idade dos falso-negativos
fn = c[c['Diagn'] == 1]['Idade']
bins_range = range(0, fn.max()+10,10)
out_fn = pd.cut(fn, bins=bins_range, include_lowest=True, right=False)

fn.hist(histtype='bar', bins=range(0, 120, 10), ec='black', zorder=2)
pl.grid(axis='x')
pl.title('Idade dos falso-negativos', fontsize=14)
pl.xlabel('Idade',fontsize=12)
pl.ylabel('Frequência', fontsize=12)
pl.xticks(range(10,120,10))
pl.savefig("./imgs/cancer/hist_falso-negativos.pdf", dpi=300)
pl.show()


# Boxplot Comparativo
pl.boxplot([fp, fn], zorder=2)
pl.xticks([1,2], ['falso-positivos', 'falso-negativos'])
pl.ylabel('Idade')
pl.title('Boxplot comparativo')
pl.grid(axis='y', zorder=0)
pl.savefig("./imgs/cancer/boxplot_comp_falso_neg_pos.pdf", dpi=300)
pl.show()

# TABELAS DE FREQUÊNCIA
freq_fp =  out_fp.value_counts(sort=False)
freq_fp = pd.DataFrame({"Freq": freq_fp})
freq_fp['Acum.'] = freq_fp['Freq'].cumsum()
freq_fp['Rel. Acum.'] = ((freq_fp['Freq']/freq_fp['Freq'].sum()).cumsum()*100).map('{:,.3f}'.format)
print()
print(freq_fp)
print()

freq_fn =  out_fn.value_counts(sort=False)
freq_fn = pd.DataFrame({"Freq": freq_fn})
freq_fn['Acum.'] = freq_fn['Freq'].cumsum()
freq_fn['Rel. Acum.'] = ((freq_fn['Freq']/freq_fn['Freq'].sum()).cumsum()*100).map('{:,.3f}'.format)
print(freq_fn)
print()

### COMPARAÇÃO DA NORMAL DAS DUAS
### FREQUENCIAS ACUMULADAS ABSULUTAS E RELATIVAS NA TAB. FREQUENCIA

#%%

sns.kdeplot(fn, label='falso-negativo', shade=1, lw=2)
sns.kdeplot(fp, label='falso-positivo', shade=1, lw=2, color='red')
pl.show()

#%%

sns.distplot(fn, hist=True, label='falso-negativo')
sns.distplot(fp, hist=True, label='falso-positivo', color='red')
pl.show()

#%%

# EXERCICIO 36 - pg. 220

# letra a) - histograma
c.LDH.hist(histtype='bar', density=True, bins=range(0,101,10), ec='black', zorder=2)
pl.grid(axis='x')
pl.title('Frequência da enzima lactate dehydrogenase (LDH)')
pl.xlabel('LDH')
pl.ylabel('Frequência')
pl.xticks(range(10,101,10))

# estatistica
mu, std = scs.norm.fit(c.LDH)

# Plot the PDF.
xmin, xmax = pl.xlim()
x = np.linspace(xmin, xmax, 100)
p = scs.norm.pdf(x, mu, std)
pl.plot(x, p, 'r--', linewidth=2)

# Teste de hipotese de normalidade com 5% de significancia:
# H0: A amostra provem de uma população normal
# H1: A amostra nao provem de uma distribuicao normal

# Testes de shapiro e lillefors: 
s   = scs.shapiro(c.LDH)
lil = lilliefors(c.LDH)

pl.text(87, 0.0722, 'Shapiro: '+str(round(s[1], 5) )+'\nLilliefors: '+str(round(lil[1], 5)), bbox=dict(facecolor='red', alpha=0.4), zorder=4 )

pl.show()

# letra a) - medidas descritivas
tab_LDH_desc = c.LDH.describe()

# letra b) 
# teste de normalidade no grafico
# teste de aderencia
# qq-plot


print(s[1], lil[1])

#%%

# EXERCICIO 41 - pg. 307

c['jovem'] = c['Idade'] <= 54

# letra a)
# Boxplot Comparativo de N
pl.boxplot([c[c['jovem'] == True]['N'], c[c['jovem'] == False]['N']])
pl.xticks([1,2], ['jovens', 'idosos'])
pl.ylabel('N')
pl.title('Boxplot comparativo -- Nitrogênio na Ureia')
pl.show()

tab_desc_N = pd.concat([c[c['jovem'] == True]['N'].describe(), c[c['jovem'] == False]['N'].describe()], axis=1).astype(float).applymap('{:,.3f}'.format)

tab_desc_N.columns = ['jovens', 'idosos']
print()
print(tab_desc_N)
print()

# letra b)
# Teste de hipotese:
#     H0: media(idosos) = 15
#     H1: media(idosos) > 15


mu_obs = c[c['jovem'] == False]['N'].mean()
mu = 15
std = 7
alpha = 0.001

z_idoso = (mu_obs - mu)/(std / len(c[c['jovem'] == False]['N'])**0.5)

z_c_idoso = scs.norm.ppf(1-alpha)
p_val_idoso =(1-scs.norm.cdf(z_idoso))

# Fazer grafico

# letra c)
# Teste de hipotese:
#     H0: media(jovem) = 15
#     H1: media(jovem) < 15

mu_obs = c[c['jovem'] == True]['N'].mean()
mu = 15
std = 5
alpha = 0.001

z_jovem = (mu_obs - mu)/(std / len(c[c['jovem'] == True]['N'])**0.5)

z_c_jovem = scs.norm.ppf(alpha)
p_val_jovem =(scs.norm.cdf(z_jovem))

# Fazer grafico

# letra d) 
# discussao do teste

#%%

# EXERCICIO 30 - pg. 368

# letra a)
doentes = c[(c['Diagn'] == 1) | (c['Diagn'] == 3)]
# doentes = doentes[doentes['N'] < 30]

pl.scatter(doentes['Idade'], doentes['N'], ec='black')
pl.ylabel("Concentração de Nitrogênio (N)")
pl.xlabel("Idade")
pl.title("Gráfico de concetração de Nitrogênio por Idade")

cor = doentes[['Idade', 'N']].corr()
# pl.text(20, 52,)

# letra b)
a1, b1, *resto1 = scs.linregress(doentes['Idade'], doentes['N'])
x_1 = np.linspace(doentes['Idade'].min(), doentes['Idade'].max(), 100)
pl.plot(x_1, a1*x_1+b1, 'r')
pl.text(80,52, r'$C = \beta I + \alpha$' +'\nC = '+str(round(a1, 3))+'I+'+str(round(b1, 3)), bbox=dict(facecolor='red', alpha=0.4))
pl.show()

# letra c)
mod = ols('Idade ~ N', data=doentes).fit()
anova_table_doentes = sm.stats.anova_lm(mod, typ=2)
print()
display(anova_table_doentes)
print()

# letra d)

nao_doentes = c[(c['Diagn'] == 2) | (c['Diagn'] == 4)]

pl.scatter(nao_doentes['Idade'], nao_doentes['N'], ec='black')
pl.ylabel("Concentração de Nitrogênio (N)")
pl.xlabel("Idade")
pl.title("Gráfico de concetração de Nitrogênio por Idade")

# letra e)
a2, b2, *resto2 = scs.linregress(nao_doentes['Idade'], nao_doentes['N'])
x_2 = np.linspace(nao_doentes['Idade'].min(), nao_doentes['Idade'].max(), 100)
pl.plot(x_2, a2*x_2+b2, 'r')
pl.text(10,40, r'$C = \beta I + \alpha$' +'\nC = '+str(round(a2, 3))+'I+'+str(round(b2, 3)), bbox=dict(facecolor='red', alpha=0.4))
pl.show()

# letra f)
mod = ols('Idade ~ N', data=nao_doentes).fit()
anova_table_na_doentes = sm.stats.anova_lm(mod, typ=2)
print()
print(anova_table_na_doentes)
print()


x_3 = np.linspace(min(doentes['Idade'].min(), nao_doentes['Idade'].min()), max(doentes['Idade'].max(), nao_doentes['Idade'].max()), 100)
pl.plot(x_3, a1*x_3+b1, 'b', label="Doentes")
pl.plot(x_3, a2*x_3+b2, 'r', label="Não doentes")
pl.legend()
pl.show()


# comp
pl.scatter(doentes['Idade'], doentes['N'], ec='black')
pl.scatter(nao_doentes['Idade'], nao_doentes['N'], ec='black')
pl.ylabel("Concentração de Nitrogênio (N)")
pl.xlabel("Idade")




#%%



#%%

