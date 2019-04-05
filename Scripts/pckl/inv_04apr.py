# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 14:19:05 2019

@author: Greta
"""

import pickle
from matplotlib import pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import os
import numpy as np

os.chdir('C:\\Users\\Greta\\Documents\\GitLab\\project\\Python_Scripts\\18Mar_RT_files\\')
    
#%%

with open('18Mar_subj_07.pkl', "rb") as fin:
    sub07 = (pickle.load(fin))[0]

#%%
os.chdir('C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Scripts\\pckl\\pckl_V1\\')
    
with open('04April_subj_07.pkl', "rb") as fin:
    sub07new = (pickle.load(fin))[0] 
    
# Sub07 and sub07new are identical. Sub07 uses the new, updated scripts.

#%%
os.chdir('C:\\Users\\Greta\\Documents\\GitHub\\ClosedLoop\\Scripts\\pckl\\')

with open('04April_V2_subj_07.pkl', "rb") as fin:
    sub07new2 = (pickle.load(fin))[0] 
    
with open('04April_V2_subj_08.pkl', "rb") as fin:
    sub08 = (pickle.load(fin))[0] 
    
with open('04April_V2_subj_11.pkl', "rb") as fin:
    sub11 = (pickle.load(fin))[0] 

with open('04April_V2_subj_13.pkl', "rb") as fin:
    sub13 = (pickle.load(fin))[0] 
    
with open('04April_V2_subj_14.pkl', "rb") as fin:
    sub14 = (pickle.load(fin))[0] 
    
with open('04April_V2_subj_17.pkl', "rb") as fin:
    sub17 = (pickle.load(fin))[0] 



#%%
# Plot showing where the correct predictions are located pr. run.

n_it=5

pred_run = np.reshape(sub11['correct_NFtest_pred'],[n_it,200])

for run in range(n_it):
    plt.figure(run)
    plt.bar(np.arange(200),pred_run[run,:])