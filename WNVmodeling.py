#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 13:15:14 2019

@author: eran
"""

''' Modelling file'''

import numpy as np
import pandas as pd
import os

directory_path='/Users/eran/Galvanize_more_repositories/WestNileVirus'

'''importing data sets:'''
train_base=pd.read_csv(os.path.join(directory_path,'train_baseline.csv'))
train_base.drop('Unnamed: 0',1,inplace=True)
sptrainW_day_of=pd.read_csv(os.path.join(directory_path,'sptrainW_day_of.csv'))
sptrainW_day_of.drop('Unnamed: 0',1,inplace=True)
sptrainW_14_days=pd.read_csv(os.path.join(directory_path,'sptrainW_14_days.csv'))
sptrainW_14_days.drop('Unnamed: 0',1,inplace=True)

'''importing models '''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

'''importing metrics '''
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score

'''importing SMOTE '''
from imblearn.over_sampling import SMOTE

'''Dicctionary of datasets'''
datasets={'baseline.imbalanced':train_base,'baseline':train_base,
          'sptrainW_day_of':sptrainW_day_of,'sptrainW_14_day':sptrainW_14_days}


'''initiating models with specific 'random' parameters:'''
log_reg=LogisticRegression(random_state=42)
knn=KNeighborsClassifier(n_neighbors=5)
rfclf=RandomForestClassifier(n_estimators=500,max_depth=2,random_state=42,max_features=8)
smt=SMOTE(random_state=42)

'''function for getting metrics for 3 different models for one dataset'''
def models_metrics(Xn,Xt,yn,yt):
    models={'log_reg':log_reg,'knn':knn,'rf':rfclf}
    ss=pd.DataFrame()
    cols=[]
    for ii in models:
        cols.append(ii)
        models[ii].fit(Xn,yn)
        ypred=models[ii].predict(Xt)
        probab=models[ii].predict_proba(Xt)
        fpr, tpr, thresholds = roc_curve(yt, probab[:,1])
        conf=confusion_matrix(ypred,yt)
        print(conf)
        TN=conf[0,0]
        FP=conf[0,1]
        FN=conf[1,0]
        TP=conf[1,1]
        print('TN',TN,'FP',FP,'FN',FN,'TP',TP)
        accur=accuracy_score(ypred,yt)
        recall=recall_score(ypred,yt)
        prec=precision_score(ypred,yt)
        aucroc=roc_auc_score(yt, probab[:,1])
        data = {'TN':TN,'FP':FP,'FN':FN,'TP':TP,'accuracy':accur,
                'recall':recall,'precision':prec,'aucroc':aucroc}
        col=pd.Series(data)
        df=pd.DataFrame(col)
        ss=pd.concat([ss,df],axis=1)
    ss.columns=cols
    return(ss)
    

'''Running script to get metrics of all datasets run through 3 models: '''
sss=pd.DataFrame()
for jj in datasets:
    Xtrain,Xtest,ytrain,ytest=train_test_split(datasets[jj].drop('WnvPresent',1),datasets[jj]['WnvPresent'],test_size=0.2,random_state=42,stratify=datasets[jj]['WnvPresent'])
    print('Xtrain',Xtrain.shape)
    print('Xtest',Xtest.shape)
    if jj!='baseline.imbalanced':
        print('smote',jj)
        Xtrain,ytrain=smt.fit_sample(Xtrain,ytrain)
    dsetdf=models_metrics(Xtrain,Xtest,ytrain,ytest)
    sss=pd.concat([sss,dsetdf],axis=1)

colsss=[list(sss.columns)[0]+'.'+list(datasets.keys())[0],list(sss.columns)[1]+'.'+list(datasets.keys())[0],list(sss.columns)[2]+'.'+list(datasets.keys())[0],
        list(sss.columns)[3]+'.'+list(datasets.keys())[1],list(sss.columns)[4]+'.'+list(datasets.keys())[1],list(sss.columns)[5]+'.'+list(datasets.keys())[1],
        list(sss.columns)[6]+'.'+list(datasets.keys())[2],list(sss.columns)[7]+'.'+list(datasets.keys())[2],list(sss.columns)[8]+'.'+list(datasets.keys())[2],
        list(sss.columns)[9]+'.'+list(datasets.keys())[3],list(sss.columns)[10]+'.'+list(datasets.keys())[3],list(sss.columns)[11]+'.'+list(datasets.keys())[3]]

sss.columns=colsss
sss



'''#################################'''

'''draft DO NOT run from this point'''

'''#################################'''
