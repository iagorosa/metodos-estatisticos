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
from statsmodels.stats.diagnostic import lilliefors


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
    
x = pd.DataFrame(lowbw[1:], columns=lowbw[0])

x = x.astype(int)

#%%

# 2.2.1. Descrição e modelagem dos dados
# a)
fn = c[c['Diagn'] == 1]['Idade']
bins_range = range(0, fn.max()+10,10)
out_fn = pd.cut(fn, bins=bins_range, include_lowest=True, right=False)



#%%
