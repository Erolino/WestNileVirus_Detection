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
sptrainW_14_days_mos=pd.read_csv(os.path.join(directory_path,'sptrainW_14_days_mos.csv'))
sptrainW_14_days_mos.drop('Unnamed: 0',1,inplace=True)


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
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

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
    ###
    maxfeat=int(np.round(np.sqrt(Xt.shape[1])))
    maxdepth=int(np.round(maxfeat/3))
    rfclf=RandomForestClassifier(n_estimators=500,max_depth=maxdepth,random_state=42,max_features=maxfeat)
    ####
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
oy=sss.T

#### ploting function for model performance depended on dataset 
#### (plotting oy)
import matplotlyb.pyplot as plt
def plt_met(mett):
    modls=['logistic Regression','K Nearest Neighbors','Random Forest Classifier']
    colorS=['b','c','g','r']
    metricss={'FN':0, 'FP':1, 'TN':2, 'TP':3, 'accuracy':4, 'aucroc':5, 'precision':6, 'recall':7}
    plt.subplots(figsize=(23,3),)
    for ai in range(3):
        plt.xticks(rotation=20,fontsize = 12)
        plt.subplot(1,3,ai+1)

        indexx = range(4)
        plt.plot(oy.iloc[[ai,ai+3,ai+6,ai+9],metricss[mett]],linestyle='None',marker='*',color=colorS[ai])
        plt.xticks(indexx, oy.index[[ai,ai+3,ai+9,ai+6]])  # set the X ticks and labels

        #plt.xticks(x, labels, rotation='vertical')
        plt.xlabel('"dataset"',fontsize = 18)
        plt.ylabel('metric: '+mett,fontsize = 18)
        plt.title(modls[ai],color=colorS[ai])

'''#################################'''

'''Comparing day-of and 14-days with hyper parameter 
    adjustments tuned for specific metrics( e.g. recall and aucroc )'''

'''#################################'''

### Grid Search
# Although Random Forest Classifier is not better than Logistic Regression 
# there are many hyper parameters that 
# could be tuned to increase its perfomance.

#Let's tune parametrs of the RF classifier towards higher recall. 
# even on the expense of lowering precision (which is currently relatively high)

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV,StratifiedKFold

rfclf2=RandomForestClassifier(random_state=42)
paramgrid={'max_depth':[2,3,5],'max_features':[4,8,16,20],
           'min_samples_split':[2,3,8],'n_estimators':[200,300,500]}
scorers={'accuracy_score':make_scorer(accuracy_score),
         'precision_score':make_scorer(precision_score),
         'recall_score':make_scorer(recall_score),
         'aucroc_score':make_scorer(roc_auc_score)}

def grid_func(the_score,jj): ## jj is the string name of the dataset
    
    #optimize hyper parametrs using grid search according to one of the scores that is input
    skf=StratifiedKFold(n_splits=5) ## stratifying the crossvalidation and choosing the number of splits=10
    GS=GridSearchCV(rfclf2,param_grid=paramgrid,scoring=scorers,refit=the_score,cv=skf,return_train_score='True',n_jobs=-1)
    
    ## choose dataset
    Xtrain,Xtest,ytrain,ytest=train_test_split(datasets[jj].drop('WnvPresent',1),datasets[jj]['WnvPresent'],test_size=0.2,random_state=42,stratify=datasets[jj]['WnvPresent'])
    Xtrain,ytrain=smt.fit_sample(Xtrain,ytrain)
    #with parallel_backend('threading'):
    GS.fit(Xtrain,ytrain)
    
    prediction=GS.predict(Xtest)
    
    print('best params for {}'.format(the_score))
    print(GS.best_params_)
    
    print('confusion matrix adjusted for max {}'.format(the_score))
    print(confusion_matrix(prediction,ytest))
    
    return(GS)
    
if 1==0:
    GSrf2=grid_func('recall_score','sptrainW_day_of') ## for day-of input the 
if 1==0:
    GSrf2eng=grid_func('recall_score','sptrainW_14_day') ## for day-of input the 
if 1==0:
    GSrfAUC=grid_func('aucroc_score','sptrainW_day_of') ## for day-of input the 
if 1==0:
    GSrfAUCeng=grid_func('aucroc_score','sptrainW_14_day') ## for day-of input the 
    

if 1==0: ## the following aggregates and presents the results from the above gridsearch objects
    result=pd.DataFrame(GSrf2.cv_results_)
    result.sort_values(['mean_test_aucroc_score','mean_test_recall_score'],inplace=True)
    result[[ 'mean_test_aucroc_score','mean_test_recall_score','mean_test_precision_score', 'mean_test_accuracy_score','param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators']].round(3).head()
    ### the best auc score place in result:
    roww0=np.where(result['mean_test_recall_score']==max(result['mean_test_recall_score']))
    ### give the winning row for highest AUC:
    winRecallresult=result.loc[roww0[0],[ 'mean_test_aucroc_score','mean_test_recall_score','mean_test_precision_score', 'mean_test_accuracy_score','param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators']]
    
    resultEng=pd.DataFrame(GSrf2eng.cv_results_)
    resultEng.sort_values(['mean_test_aucroc_score','mean_test_recall_score'],inplace=True)
    resultEng[[ 'mean_test_aucroc_score','mean_test_recall_score','mean_test_precision_score', 'mean_test_accuracy_score','param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators']].round(3).head()
    ### the best auc score place in result:
    roww1=np.where(resultEng['mean_test_recall_score']==max(resultEng['mean_test_recall_score']))
    ### give the winning row for highest AUC:
    winRecallresultEng=resultEng.loc[roww1[0],[ 'mean_test_aucroc_score','mean_test_recall_score','mean_test_precision_score', 'mean_test_accuracy_score','param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators']]
    
    resultAuc=pd.DataFrame(GSrfAUC.cv_results_)
    resultAuc.sort_values(['mean_test_aucroc_score','mean_test_recall_score'],inplace=True)
    resultAuc[[ 'mean_test_aucroc_score','mean_test_recall_score','mean_test_precision_score', 'mean_test_accuracy_score','param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators']].round(3).head()
    ### the best auc score place in result:
    roww2=np.where(resultAuc['mean_test_aucroc_score']==max(resultAuc['mean_test_aucroc_score']))
    ### give the winning row for highest AUC:
    winAucresult=resultAuc.loc[roww2[0],[ 'mean_test_aucroc_score','mean_test_recall_score','mean_test_precision_score', 'mean_test_accuracy_score','param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators']]
    
    resultEngAuc=pd.DataFrame(GSrfAUCeng.cv_results_)
    resultEngAuc.sort_values(['mean_test_aucroc_score','mean_test_recall_score'],inplace=True)
    resultEngAuc[[ 'mean_test_aucroc_score','mean_test_recall_score','mean_test_precision_score', 'mean_test_accuracy_score','param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators']].round(3).head()
    ### the best auc score place in result:
    roww3=np.where(resultEngAuc['mean_test_aucroc_score']==max(resultEngAuc['mean_test_aucroc_score']))
    ### give the winning row for highest AUC:
    winAucresultEng=resultEngAuc.loc[roww3[0],[ 'mean_test_aucroc_score','mean_test_recall_score','mean_test_precision_score', 'mean_test_accuracy_score','param_max_depth', 'param_max_features', 'param_min_samples_split', 'param_n_estimators']]

'''#################################'''

'''Running final model, using plotting function to summarize'''

'''#################################'''


'''Let's use these 'winning' parameters to run a RF model 
again, for further data analysis '''
## winAUCresult[['param_max_depth','param_max_features','param_min_samples_split','param_n_estimators']]

# lat's put them manualy:
win_params={'n_estimators':500,'min_samples_split':3,'max_features':20,'max_depth':5}

rfwin=RandomForestClassifier(random_state=42,n_estimators=win_params['n_estimators'],
                             min_samples_split=win_params['min_samples_split'],
                             max_features=win_params['max_features'],
                             max_depth=win_params['max_depth'])
rfwin.fit(Xtrain,ytrain)
predy=rfwin.predict(Xtest)
predprob=rfwin.predict_proba(Xtest)
fpr, tpr, thresholds = roc_curve(ytest, predprob[:,1])
precision, recall, threshol=precision_recall_curve(ytest, predprob[:,1])

def plt_curvs(ytest, predprob):
    ###
    fpr, tpr, thresholds = roc_curve(ytest, predprob[:,1])
    precision, recall, threshol=precision_recall_curve(ytest, predprob[:,1])
    ###
    print("roc_auc_score: ",round(roc_auc_score(ytest, predprob[:,1]),3))
    print("precision-recall_auc_score: ",round(auc(recall,precision),3))
    plt.subplots(figsize=(16,3))
    plt.subplot(1,3,1)
    xx=np.linspace(0,1,1000)
    yy=xx
    plt.plot(xx,yy,'--')
    plt.plot(fpr,tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #
    plt.subplot(1,3,2)
    plt.plot([0,1],[0.01,0.01],'--')
    plt.plot(recall,precision,color='orange')
    plt.xlabel('recall')
    j=plt.ylabel('precision')
    #
    plt.subplot(1,3,3)
    plt.plot(threshol,precision[:-1],'gold',label="Precision")
    plt.plot(threshol,recall[:-1],'purple',label='Recall')
    plt.xlabel('Thresholds')
    j=plt.ylabel('Score')
    plt.legend(loc='best')

'''#################################'''

'''draft DO NOT run from this point'''

'''#################################'''

''' The following function is number of mosquitos prediction section '''

''' Pseudo Code: 
    
    import regression tree model
    regtree=regression_tree()
    regtree.fit()
    
    Change the following grid_func for regression:'''

if 1==0:
    def grid_func_reg(the_score,jj,target): ## jj is the string name of the dataset
        
        #optimize hyper parametrs using grid search according to one of the scores that is input
        skf=StratifiedKFold(n_splits=5) ## stratifying the crossvalidation and choosing the number of splits=10
        GS=GridSearchCV(rfclf2,param_grid=paramgrid,scoring=scorers,refit=the_score,cv=skf,return_train_score='True',n_jobs=-1)
        
        ## choose dataset
        Xtrain,Xtest,ytrain,ytest=train_test_split(datasets[jj].drop(target,1),datasets[jj][target],test_size=0.2,random_state=42,stratify=datasets[jj][target])
        Xtrain,ytrain=smt.fit_sample(Xtrain,ytrain)
        #with parallel_backend('threading'):
        GS.fit(Xtrain,ytrain)
        
        prediction=GS.predict(Xtest)
        
        print('best params for {}'.format(the_score))
        print(GS.best_params_)
        
        print('confusion matrix adjusted for max {}'.format(the_score))
        print(confusion_matrix(prediction,ytest))
        
        return(GS)