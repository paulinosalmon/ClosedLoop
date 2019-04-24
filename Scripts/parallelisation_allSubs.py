#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 12:47:28 2018

@author: jonf
"""
import sys
import numpy as np
import pandas as pd
import os
import pickle
import multiprocessing
import time

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection, linear_model
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import zscore

from offline_analysis_FUNC import *

multiprocessing.cpu_count()


class p:
    # =============================================================================
    # Default setting
    # =============================================================================
    nCors                   = 22
    shouldCheckFirst        = True


def fit(subjID):    
    print('Starting fitting aka analyzeOffline for subjID '+str(subjID))
    analyzeOffline(subjID)
    print('Fitting done aka analyzeOffline')
    

def train():
    sub_types = ['07','08','11','13','14','15','16','17','18','19',\
          '21','22','23','24','25','26','27','30','31','32','33','34']
    
    processes   = [] # for parallelism
    
    for subjID in sub_types:
        proc = multiprocessing.Process(target=fit, args=(subjID,))
        proc.start()
        processes.append(proc)
        print('len of processes: ' + str(len(processes)))
        
        # Hold until processes are done if exceeding no. cores
        if len(processes) >= p.nCors:
            for pr in processes:
                pr.join()
            processes   = []
    
    # End all processes
#    for pr in processes:
#        pr.join()
#        


        
#%%

if __name__ == '__main__':
    train()