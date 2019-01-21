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

'''##########################################################'''

''' TRAIN + SPRAY PART - CLEAN, ENGINEER AND MERGE'''

'''##########################################################'''

# create dummies for species
train1=pd.get_dummies(train,columns=['Species'],drop_first=True)    
train1['Date']=pd.to_datetime(train1['Date'])

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
spray2=spray1.groupby(['Longitude','Latitude']).Date.agg(['count','min','max',np.ptp]).reset_index() ## get the range of spraying for each site
spray2=spray2.sort_values('ptp',ascending=False)
#merge train with spray:
sptrainright=pd.merge(spray2,train1,on=['Longitude','Latitude'],how='right',indicator=True) ## merge train with spray 
sptrainright['Date']=pd.to_datetime(sptrainright['Date']) 
sptrainright['delmin']=sptrainright['Date']-sptrainright['min']
sptrainright['delmin']=sptrainright['delmin'].dt.days
sptrainright['delmax']=sptrainright['Date']-sptrainright['max']
sptrainright['delmax']=sptrainright['delmax'].dt.days
sptrainright['most_recent_spray_(days)']=3650
## previously to create a column with the most recent spray in days
# after 962 obdservations with spray, the rest is without data on spray
# Make an inner merge to concentrate on the spray&train intersection (and avoid dealing with NaT and NaN:
sptraininner=pd.merge(spray2,train1,on=['Longitude','Latitude'],how='inner',indicator=True)    
sptraininner['Date']=pd.to_datetime(sptraininner['Date'])
sptraininner['delmin']=sptraininner['Date']-sptraininner['min']
sptraininner['delmin']=sptraininner['delmin'].dt.days
sptraininner['delmax']=sptraininner['Date']-sptraininner['max']
sptraininner['delmax']=sptraininner['delmax'].dt.days
sptraininner['most_recent_spray_(days)']=sptraininner.apply(lambda x: recent(x['delmin'],x['delmax']),axis=1)
# Now conocat the dfs
sptrain=pd.concat([sptraininner,sptrainright.loc[963:,:]])
# arrange columns name
'''
colls=sptrain.columns
colis=list(colls)
colsnew=colis[:2]+colis[3:5]+[colis[-1]]+colis[6:14]+colis[15:21]+colis[14:15] 
sptrain=sptrain[colsnew]
'''
sptrain.rename(columns={'Date':'Date_of_collection'},inplace=True)
sptrain.drop(['_merge'],1,inplace=True)
# let' add a column - whether the area was recently sprayed (i.e. <150 days)
sptrain['Recently_sprayed']=(sptrain['most_recent_spray_(days)']<150).astype(int)
sptrain['Recently_sprayed'].value_counts()

'''##########################################################'''

''' WEATHER PART - CLEAN, ENGINEER AND MERGE'''

'''##########################################################'''

weather_csv=os.path.join(directory_path,"all/weather.csv")
weather=pd.read_csv(weather_csv)

## Create weather station column in the train data (2 different stations: 1 and 2)
## first create a function to get classify which weather station fits which trap collection (by geo-location)
# assumption: long and lat have the same resolution when converted to distance.
# for distance calculation using: distance^2=long^2+lat^2

import random
def stationing(long,lat):
    sta1={'long':-87.933,'lat': 41.995}
    sta2={'long':-87.752,'lat': 41.786}
    longdelt1=long-sta1['long']
    latdelt1=lat-sta1['lat']
    dist1 = np.sqrt((longdelt1)**2+(latdelt1)**2)
    #print("dist1 =",dist1)
    longdelt2=long-sta2['long']
    latdelt2=lat-sta2['lat']
    dist2 = np.sqrt((longdelt2)**2+(latdelt2)**2)
    #print("dist2 =",dist2)
    if dist1<dist2:
        #print("stat1",dist1)
        station=1
    elif dist1>dist2:
        station=2
        #print("stat2",dist2)
    else:
        station=random.choice([1,2])
    return(station)
    
sptrainW=sptrain.copy() # make a copy of sptrainW so we can add weather (W) features to it. (starting with station column) 
sptrainW['station']=sptrainW.apply(lambda x: stationing(x.Longitude,x.Latitude),axis=1)  
## using stationing function to match station number to every observation
## now we have station column. ~80% of the traps are closer to station 2
weather['Date']=pd.to_datetime(weather['Date'])

# merging weather and sptrain data 
sptrainW0=pd.merge(sptrainW,weather,left_on=['Date_of_collection','station'],right_on=['Date','Station'],how='left',indicator=True)
## sptrainW0 is the merged dataset before any engineering and cleanning of it
sptrainW1=sptrainW0.copy()
## let's start engineering:
sptrainW1.drop(['station','Date'],1,inplace=True)
sptrainW1.drop(['_merge',],1,inplace=True)
sptrainW1.drop(['Depart',],1,inplace=True)# Depart - # Most is M missing (8223), so droping that column
#Since number of missing values are small (26), mode or median would make sense as replacement. 
# But after doing quick search online, we can approximate WetBulb from DewPoint and 
# temperature (that we have) with this formula - TAVG-((TAVG-DEWPOINT)/3).
# Writing a function for wetbulb approximation:
def wetbulb(tavg,dp,wb):
    if wb=='M':
        wb=tavg-(tavg-dp)/3
    else: 
        pass
    return(wb)
  
sptrainW1['Tavg']=sptrainW1['Tavg'].astype(int) ## numbers are stored as str - so turn to int, to manipulate 
# applying it to the df:  
sptrainW1['WetBulb']=sptrainW1.apply(lambda x: wetbulb(x['Tavg'],x['DewPoint'],x['WetBulb']),axis=1)
## continuing switching str into int in other columns
sptrainW1['Heat']=sptrainW1['Heat'].astype(int)
sptrainW1['Cool']=sptrainW1['Cool'].astype(int)
sptrainW1.drop(['Sunrise','Sunset'],1,inplace=True) ## mostly empty
#Function to turn codes into 2 groups good (' ') and bad weather ( all other codes)
def codes(col):
    if col==' ':
        col='Norm'
    else:
        col='Bad'
    return(col)
codes(' ')

sptrainW1['weather_type']=sptrainW1.apply(lambda x: codes(x['CodeSum']), axis=1)
sptrainW1=pd.get_dummies(sptrainW1,columns=['weather_type'],drop_first=True)
# to check: type sptrainW1.weather_type_Norm.value_counts()
sptrainW1.drop(['Water1','SnowFall'],1,inplace=True) ## see summary, mostly missing values
# PrecipTotal - convert T (trace) to 0.005 (look at summary):
sptrainW1['PrecipTotal']=sptrainW1['PrecipTotal'].apply(lambda x: 0.005 if x=='  T' else x)
#convert 'M' to mode 
import statistics as st
mode=st.mode(sptrainW1['PrecipTotal']) # mode is '0'
sptrainW1['PrecipTotal']=sptrainW1['PrecipTotal'].apply(lambda x: mode if x=='M' else x)
sptrainW1['PrecipTotal']=sptrainW1['PrecipTotal'].astype(float) # converting to type float.
 # Depth:
sptrainW1.drop('Depth',inplace=True,axis=1) # drop, mostly 'M' rest 0 (see summary)
# Stn Pressure:
moud=st.mode(sptrainW1['StnPressure'])
sptrainW1['StnPressure']=sptrainW1['StnPressure'].apply(lambda x: moud if x=='M' else x)
sptrainW1['StnPressure']=sptrainW1['StnPressure'].astype(float)
# Sealevel
sptrainW1['SeaLevel']=sptrainW1['SeaLevel'].astype(float)
