#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 14:11:00 2019

@author: eran
"""

import numpy as np
import pandas as pd
import os

directory_path='/Users/eran/Galvanize_more_repositories/WestNileVirus'

train=pd.read_csv(os.path.join(directory_path,'all/train.csv'))


''' pseudo code:
def engineerTR(train):'''
train1=pd.get_dummies(train,columns=['Species'],drop_first=True)    
train1['Date']=pd.to_datetime(train1['Date'])
        #
    #    return(clean_merged)
    #save

# Let's create a function that finds the days since the most recent spray, 
# given a date of collection minus dates of spraying:
def recent(delMin,delMax):
    if (delMin>=0) & (delMax<0):
        dell=delMin
    elif (delMax>=0) & (delMin<0):
        dell=delMax
    elif (delMax<0) & (delMin<0):
        dell=3650
    elif (delMax>=0) & (delMin>=0):
        if delMax<delMin:
            dell=delMax
        elif delMax>delMin:
            dell=delMin
        else:
            dell=delMin
    elif delMin.isnull() or delMax.isnull():
        dell=3650
    return(dell)
#def engineerSP(spray):
 
spray=pd.read_csv(os.path.join(directory_path,'all/spray.csv'))
spray1=spray.copy()
spray1['Longitude']=spray['Longitude'].round(3) ## rounding to match longitudes in train data (resolution of 100 M)
spray1['Latitude']=spray['Latitude'].round(3)
train1['Longitude']=train1['Longitude'].round(3)
train1['Latitude']=train1['Latitude'].round(3)
spray1['Date']=pd.to_datetime(spray1['Date']) ## change to datetime object so operations could be done 
spray2=spray1.groupby(['Longitude','Latitude']).Date.agg(['count','min','max',np.ptp]).reset_index()
spray2=spray2.sort_values('ptp',ascending=False)
#merge train with spray:
sptrainright=pd.merge(spray2,train1,on=['Longitude','Latitude'],how='right',indicator=True) ## merge train with spray 
sptrainright['Date']=pd.to_datetime(sptrainright['Date']) 
sptrainright['delmin']=sptrainright['Date']-sptrainright['min']
sptrainright['delmin']=sptrainright['delmin'].dt.days
sptrainright['delmax']=sptrainright['Date']-sptrainright['max']
sptrainright['delmax']=sptrainright['delmax'].dt.days
sptrainright['most_recent_spray_(days)']=sptrainright.apply(lambda x: recent(x['delmin'],x['delmax']),axis=1) ## use the ad hoc function writen previously
# after 962 obdservations with spray, the rest is without data on spray
# Make an inner merge to concentrate on the spray&train intersection (and avoid dealing with NaT and NaN:
sptraininner=pd.merge(spray2,train1,on=['Longitude','Latitude'],how='inner',indicator=True)    
sptraininner['Date']=pd.to_datetime(sptraininner['Date'])
sptraininner['delmin']=sptraininner['Date']-sptraininner['min']
sptraininner['delmin']=sptraininner['delmin'].dt.days
sptraininner['delmax']=sptraininner['Date']-sptraininner['max']
sptraininner['delmax']=sptraininner['delmax'].dt.days
sptraininner['most_recent_spray (days)']=sptraininner.apply(lambda x: recent(x['delmin'],x['delmax']),axis=1)
sptraininner['most_recent_spray (days)']=3650
# Now conocat the dfs
sptrain=pd.concat([sptraininner,sptrainright.loc[963:,:]])
# arrange columns name
colls=sptrain.columns
colis=list(colls)
colsnew=colis[:2]+colis[3:5]+[colis[-1]]+colis[6:14]+colis[15:21]+colis[14:15] 
sptrain=sptrain[colsnew]
sptrain.rename(columns={'Date':'Date_of_collection'},inplace=True)
# let' add a column - whether the area was recently sprayed (i.e. <150 days)
sptrain['Recently_sprayed']=(sptrain['most_recent_spray (days)']<150).astype(int)
sptrain['Recently_sprayed'].value_counts()


def engineerWE(weather):
        
    def merge_all():
        
        merge
        splt back to: train and test
    
